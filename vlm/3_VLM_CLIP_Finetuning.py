# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: conda-base-py
# ---

# + [markdown] id="cd5ec949-8b86-4a7f-b031-01767c573e5d"
# # Fine-tuning CLIP
# Run the following cell if running this Jupyter Notebook on Google Colab to install additional necessary libraries before you begin. If you are running this on your Vertex AI Workbench Instance, you will likely already have installed these libraries.

# + id="61bef171-f4f0-404a-b459-b71df7967663"
# %%capture
# for google Colab
# !pip install accelerate transformers==4.37.0 datasets
# !pip install --upgrade --q datasets transformers accelerate soundfile librosa evaluate jiwer tensorboard gradio

# + [markdown] id="1897940c-1da6-4451-85aa-2b5a32c11580"
# ## Initialize CLIP Model
# Here we initialize the CLIP model as well as a particular tokenizer; here we've chosen the RoBERTa tokenizer.

# + id="13e72766-d808-4418-8bfb-a6b174783193" outputId="d4a67571-2915-4775-e9bc-82bbfaa8b22d"
import torch
from datasets import load_dataset
from PIL import Image
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    Trainer,
    TrainingArguments,
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    AutoTokenizer,
    AutoImageProcessor
)

from typing import List
import io
from transformers import AutoProcessor, Owlv2ForObjectDetection
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
name = "google/owlv2-base-patch16-ensemble"
processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
model.to(device)
image_processor = AutoImageProcessor.from_pretrained(name)
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")

config = model.config
image_processor

# + [markdown] id="c22d6e8e-72ac-4ac9-9d91-9227f2fe6622"
# Now we load our datasets. Here we're loading a small dummy COCO dataset.

# + [markdown] id="e14acade"
# # Example of data (image, text pairs)

# +
import jsonlines
import torchaudio
from datasets import Dataset, load_metric, DatasetDict
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Trainer, TrainingArguments
from pathlib import Path
import torch
import librosa
import IPython.display as ipd
import jiwer

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Define the path to the directory
current_directory = Path.cwd()
file_path = current_directory / '..' / '..' / 'novice'
data_dir = file_path.resolve()
print(data_dir, current_directory)

# +
import jsonlines
import torchaudio
from datasets import Dataset, load_metric, DatasetDict
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Trainer, TrainingArguments
from pathlib import Path
import torch
import librosa
import IPython.display as ipd
import jiwer

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Define the path to the directory
current_directory = Path.cwd()
file_path = current_directory / '..' / '..' / 'novice'
data_dir = file_path.resolve()
print(data_dir, current_directory)

# Read data from a jsonl file and reformat it
data = {'key': [], 'image': [], 'caption': [], 'bbox': []}
counter = 0
with jsonlines.open(data_dir / "vlm.jsonl") as reader:
    for i, obj in enumerate(reader):
        if len(data['image']) < 10:
            for item in obj['annotations']:
                data['key'].append(counter)
                data['caption'].append(item['caption'])
                data['image'].append(obj['image'])
                data['bbox'].append(item['bbox'])
                counter += 1
                
# Convert to a Hugging Face dataset
dataset = Dataset.from_dict(data) # converts it into a dataset object which has in-built helper functions to help us later on when we need to do operations on it
# think of it as a special pandas library :)

# Shuffle the dataset
dataset = dataset.shuffle(seed=42) # shuffle the dataset (one of the in-built helper functions of the Hugging Face dataset)

# Split the dataset into training, validation, and test sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
print(train_size, val_size, test_size)

train_dataset = dataset.select(range(train_size))
val_dataset = dataset.select(range(train_size, train_size + val_size))
test_dataset = dataset.select(range(train_size + val_size, train_size + val_size + test_size))

dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset,
    'val': val_dataset})

dataset
# -

dataset['train'][0]


# + [markdown] id="1b91ff5c"
# # Preprocess the data

# + [markdown] id="e80c4798-5f10-4afb-831f-f96195490eb7"
# We need to pre-process our dataset such that our model will be able to recognize it. So first we define our image preprocessing logic (e.g. resizing, converting to the correct datatype, normalization, etc.), as well as our text preprocessing logic (i.e. tokenization), then apply it to our datasets, both train and eval.

# + id="04a6deb7-9456-4898-b832-94f85f4ecb77"
# We use torchvision for faster image pre-processing. The transforms are implemented as nn.Module,
# so we jit it to be faster.
class Transform(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize([image_size], interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
        )

    def forward(self, x) -> torch.Tensor:
        """`x` should be an instance of `PIL.Image.Image`"""
        with torch.no_grad():
            x = self.transforms(x)
        return x

# For preprocessing the datasets.
# Initialize torchvision transforms and jit it for faster processing.
image_transformations = Transform(
    config.vision_config.image_size, image_processor.image_mean, image_processor.image_std
)
image_transformations = torch.jit.script(image_transformations)


# + id="158017bd-106e-49b2-9ec6-a020ef6a134b"
def preprocess_dataset(dataset, split):
    # Preprocessing the datasets.
    data = dataset[split]
    # We need to tokenize inputs and targets.
    column_names = data.column_names

    # 6. Get the column names for input/target.
    image_column = "image_path"
    caption_column = "caption"
    dataset_columns = (image_column, caption_column)

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples):
        captions = list(examples[caption_column])
        text_inputs = tokenizer(captions, padding="max_length", truncation=True)
        examples["input_ids"] = text_inputs.input_ids
        examples["attention_mask"] = text_inputs.attention_mask
        return examples

    def transform_images(examples):
        images = [read_image(image_file, mode=ImageReadMode.RGB) for image_file in examples[image_column]]
        examples["pixel_values"] = [image_transformations(image) for image in images]
        return examples

    data = data.map(
        function=tokenize_captions,
        batched=True,
        remove_columns=[col for col in column_names if col != image_column],
        desc=f"Running tokenizer on {split} dataset",
    )

    # Transform images on the fly as doing it on the whole dataset takes too much time.
    data.set_transform(transform_images)
    return data


# + colab={"referenced_widgets": ["923444efb27644339598971bc38e4d3e"]} id="41c47421-96d7-442d-8ca9-71a87adb088a" outputId="0abd5785-8c48-4df5-fdf0-8acc7f662ff5"
train_dataset = preprocess_dataset(dataset, "train")
eval_dataset = preprocess_dataset(dataset, "val")


# + [markdown] id="23a81369-f7ed-4e4b-8477-7c38bf29f71b"
# Finally we need to write a small function to handle the batching logic for our training. This collates all passed training items in the batch together such that we can pass it to the model for training, along with the kwarg `return_loss=True` such that the model will return its loss for backpropagation.

# + id="696678d7-2869-4a02-b3c5-481bd6419b69"
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "return_loss": True,
    }


# + [markdown] id="4ce09610"
# # Train

# + [markdown] id="91d0a242-1823-4515-850f-19d9bb2ae2c8"
# Now we're ready to actually train our CLIP model!

# + id="8fb72d8d-7d65-4a65-9954-ae4b12d71911" outputId="bf987e9d-3f90-4977-b427-0801ab2a40f2"
# initialize Trainer
training_args = TrainingArguments(
    learning_rate=5e-5,
    warmup_steps=0,
    weight_decay=0.1,
    per_device_train_batch_size=16,
    logging_steps=5,
    save_steps=5,
    remove_unused_columns=False,
    output_dir="clip-finetune",
    report_to='none', # disable wandb
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
)
train_result = trainer.train()

# + id="ec776e85-d358-4320-ac1a-3cc746747853" outputId="9153542b-f723-479a-e00d-9ff298709bb9"
metrics = trainer.evaluate()
print(metrics)

# + [markdown] id="6e4b588b-1ce1-4007-8682-3fa244a72ec2"
# Once the model is trained, we can save it to our defined `output_dir` (in this case `clip-finetune`) so we can import it into our applications later.

# + id="b5e68105-e4ed-481c-bc7c-57ba2179b7ff" outputId="2aa55481-f703-4a47-c663-69f5e52a5e93"
trainer.save_model("clip-finetune")
tokenizer.save_pretrained("clip-finetune")
image_processor.save_pretrained("clip-finetune")

# + [markdown] id="22cc07fb-87e0-4898-8c87-df2ac1a1d5fb"
# ## Resources
# * [HF Transformers on training CLIP](https://github.com/huggingface/transformers/tree/main/examples/pytorch/contrastive-image-text)
