#!/usr/bin/env python3
"""
GPT Image Classification Script

This script uses OpenAI's GPT models to classify images into predefined categories.
It supports different datasets, test configurations, and includes a caching system
to avoid redundant API calls.

Usage:
    python execute_gpt.py --models gpt-4o gpt-4o-mini --datasets ArtDL IconArt --folders test_1 test_2
    
Features:
    - Supports multiple GPT models (gpt-4o, gpt-4o-mini, etc.)
    - Handles different datasets and test configurations
    - Implements caching to save API calls and costs
    - Provides cost estimation for API usage
    - Supports few-shot learning with example images
"""

import os
import click
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import openai

# Increase PIL's DecompressionBombWarning threshold to ~200 million pixels
Image.MAX_IMAGE_PIXELS = 200000000
import base64
import json
import logging
from configparser import ConfigParser
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# Import custom logger
import logger_utils


class ModelConfig:
  """Configuration for different GPT models including pricing."""

  MODELS = {
      "gpt-4o": {
          "input_cost": 2.5,   # Cost per 1M input tokens
          "output_cost": 10.0  # Cost per 1M output tokens
      },
      "gpt-4o-mini": {
          "input_cost": 0.150,  # Cost per 1M input tokens
          "output_cost": 0.600  # Cost per 1M output tokens
      }
  }

  @classmethod
  def get_costs(cls, model: str) -> Tuple[float, float]:
    """
    Get the input and output token costs for a specific model.

    Args:
        model: The model name (e.g., 'gpt-4o')

    Returns:
        Tuple of (input_cost, output_cost) per 1M tokens

    Raises:
        ValueError: If the model is not supported
    """
    if model not in cls.MODELS:
      raise ValueError(
        f"Unsupported model: {model}. Available models: {list(cls.MODELS.keys())}")

    config = cls.MODELS[model]
    return config["input_cost"], config["output_cost"]


class CacheManager:
  """Manages caching of API responses to avoid redundant calls."""

  def __init__(self, base_dir: str, dataset: str, test: str, model: str, ignore_zero_cache: bool = False, logger=None):
    """
    Initialize the cache manager.

    Args:
        base_dir: Base directory for the project
        dataset: Dataset name (e.g., 'ArtDL')
        test: Test identifier (e.g., 'test_1')
        model: Model name (e.g., 'gpt-4o')
        ignore_zero_cache: Whether to ignore zero arrays in cache
        logger: Logger instance
    """
    self.cache_dir = os.path.join(base_dir, 'gpt_data', 'cache')
    os.makedirs(self.cache_dir, exist_ok=True)

    # Create directories for the dataset and test
    os.makedirs(os.path.join(self.cache_dir, dataset), exist_ok=True)
    os.makedirs(os.path.join(self.cache_dir, dataset, test), exist_ok=True)

    # For backward compatibility, use the old cache file format
    self.cache_file = os.path.join(os.path.join(
      self.cache_dir, dataset, test), f'{model}.json')

    self.metadata = {
        "model": model,
        "dataset": dataset,
        "test": test,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0"
    }

    self.ignore_zero_cache = ignore_zero_cache
    self.logger = logger or logging.getLogger("default")
    self.cache = self._load_cache()

  def _load_cache(self) -> Dict[str, List[float]]:
    """
    Load the cache from disk.

    Returns:
        Dictionary mapping image IDs to probability arrays
    """
    if os.path.exists(self.cache_file):
      with open(self.cache_file, 'r') as file:
        try:
          cache = json.load(file)

          # Only filter out zero arrays if ignore_zero_cache is enabled
          if self.ignore_zero_cache:
            valid_cache = {k: v for k, v in cache.items()
                           if isinstance(v, list) and len(v) > 0 and not all(x == 0 for x in v)}
            self.logger.info(
              f"Loaded {len(valid_cache)} valid cached results (ignoring zero arrays)")
            return valid_cache
          else:
            valid_cache = {k: v for k, v in cache.items()
                           if isinstance(v, list) and len(v) > 0}
            self.logger.info(f"Loaded {len(valid_cache)} cached results")
            return valid_cache
        except json.JSONDecodeError:
          self.logger.warning(
            f"Cache file {self.cache_file} is corrupted. Creating new cache.")
          return {}
    return {}

  def get_result(self, image_id: str) -> Optional[List[float]]:
    """
    Get cached result for an image.

    Args:
        image_id: The image identifier

    Returns:
        List of probabilities or None if not in cache
    """
    result = self.cache.get(image_id)

    # If ignore_zero_cache is True, treat all-zero arrays as cache misses
    if self.ignore_zero_cache and result is not None and all(x == 0 for x in result):
      return None

    return result

  def add_result(self, image_id: str, probabilities: List[float]) -> None:
    """
    Add a new result to the cache.

    Args:
        image_id: The image identifier
        probabilities: List of class probabilities
    """
    self.cache[image_id] = probabilities

  def save(self, periodic: bool = False, batch_count: int = 0, save_frequency: int = 5) -> None:
    """
    Save cache to disk.

    Args:
        periodic: Whether this is a periodic save
        batch_count: Current batch count (for periodic saves)
        save_frequency: How often to save (in batches)
    """
    if not periodic or (batch_count % save_frequency == 0):
      with open(self.cache_file, 'w') as file:
        json.dump(self.cache, file, indent=4)

      if periodic:
        self.logger.info(f"Cache saved after {batch_count} batches")


class GPTImageClassifier:
  """Classifies images using OpenAI's GPT models with vision capabilities."""

  def __init__(self, model: str, api_key: str, dataset: str, test: str, base_dir: str, ignore_zero_cache: bool = False, logger=None):
    """
    Initialize the classifier.

    Args:
        model: The GPT model to use (e.g., 'gpt-4o')
        api_key: OpenAI API key
        dataset: Dataset name
        test: Test identifier
        base_dir: Base directory for the project
        ignore_zero_cache: Whether to ignore zero arrays in cache
        logger: Logger instance
    """
    self.model = model
    self.client = openai.Client(api_key=api_key)
    self.dataset = dataset
    self.test = test
    self.base_dir = base_dir

    # Set up logger
    self.logger = logger or logging.getLogger("default")

    # Initialize cache manager
    self.cache_manager = CacheManager(
      base_dir, dataset, test, model, ignore_zero_cache=ignore_zero_cache, logger=self.logger)

    # Initialize prompt folder
    self.prompt_folder = os.path.join(base_dir, os.pardir, 'prompts')

    # Token usage tracking
    self.total_input_tokens = 0
    self.total_output_tokens = 0

  def _get_prompt_path(self, dataset: str, test: str) -> str:
    """
    Get the path to the appropriate prompt file.

    Args:
        dataset: Dataset name
        test: Test identifier

    Returns:
        Path to the prompt file
    """
    # Determine system prompt
    prompt_dataset_folder = os.path.join(self.prompt_folder, dataset)
    if not os.path.exists(prompt_dataset_folder):
      raise FileNotFoundError(
        f"Prompt folder does not exist: {prompt_dataset_folder}")

    return os.path.join(prompt_dataset_folder, f'{test}.txt')

  def _load_few_shot_examples(self, dataset: str) -> List[Dict[str, Any]]:
    """
    Load few-shot examples for the specified dataset.

    Args:
        dataset: Dataset name

    Returns:
        List of message dictionaries for few-shot examples
    """
    few_shot_messages = []

    few_shot_folder = os.path.join(
      self.base_dir, os.pardir, 'dataset', f"{dataset}-data", 'few-shot')
    few_shot_file = os.path.join(few_shot_folder, 'train_data.csv')

    if not os.path.exists(few_shot_file):
      return few_shot_messages

    few_shot_df = pd.read_csv(few_shot_file)

    # Get class descriptions for better few-shot examples
    classes_df = pd.read_csv(os.path.join(
      self.base_dir, os.pardir, 'dataset', f"{dataset}-data", 'classes.csv'))
    class_descriptions = dict(zip(classes_df['ID'], classes_df['Description']))

    for _, row in few_shot_df.iterrows():
      image_path = os.path.join(few_shot_folder, f'{row["item"]}.jpg')

      if not os.path.exists(image_path):
        continue

      # Enhanced user message with clear instruction
      few_shot_messages.append(
          {"role": "user", "content": [
              {"type": "text", "text": "Please classify this image into one of the provided categories."},
              {"type": "image_url", "image_url": {
                "url": encode_image(image_path)}}
          ]},
      )

      # Enhanced assistant response with reasoning
      class_id = row['class']
      class_desc = class_descriptions.get(class_id, "")

      # Create a more detailed response with reasoning
      assistant_response = f"This image depicts {class_id}"
      if class_desc:
        assistant_response += f" ({class_desc})"
      assistant_response += "."

      few_shot_messages.append(
          {"role": "assistant", "content": assistant_response}
      )

    return few_shot_messages

  def _prepare_batch_request(self, batch: List[Tuple[str, str]], system_prompt: str, few_shot_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prepare the API request for a batch of images.

    Args:
        batch: List of (item_id, image_path) tuples
        system_prompt: The system prompt to use
        few_shot_messages: Few-shot example messages

    Returns:
        List of message dictionaries for the API request
    """
    content = [
        {"type": "text", "text": "Please classify the following set of images:"}
    ]

    for item, image_path in batch:
      content.append({"type": "text", "text": f"Image (ID: {item}):"})
      content.append({"type": "image_url", "image_url": {
                     "url": encode_image(image_path)}})

    messages = [{"role": "system", "content": system_prompt}] + \
        few_shot_messages + [{"role": "user", "content": content}]
    return messages

  def _find_similar_class(self, predicted_class: str, classes: List[Tuple[str, str]]) -> Optional[str]:
    """
    Find a similar class ID if the predicted class doesn't match exactly.
    
    Args:
        predicted_class: The predicted class ID from the model
        classes: List of (class_id, class_description) tuples
        
    Returns:
        The matching class ID if found, None otherwise
    """
    # If the predicted class is already a valid class ID, return it
    for cls_id, _ in classes:
      if predicted_class == cls_id:
        return cls_id
    
    # Check for similar class IDs (e.g., "11H(MARY)" vs "11F(MARY)")
    # Extract the base part and the description part
    import re
    match = re.match(r'([0-9]+[A-Za-z]*)(?:\(([^)]+)\))?', predicted_class)
    if match:
      base_part = match.group(1)
      desc_part = match.group(2)
      
      # Look for classes with the same description part
      if desc_part:
        for cls_id, _ in classes:
          cls_match = re.match(r'([0-9]+[A-Za-z]*)(?:\(([^)]+)\))?', cls_id)
          if cls_match and cls_match.group(2) == desc_part:
            self.logger.info(f"Reconciled similar class: {predicted_class} -> {cls_id}")
            return cls_id
      
      # Look for classes with the same base part
      for cls_id, _ in classes:
        cls_match = re.match(r'([0-9]+[A-Za-z]*)(?:\(([^)]+)\))?', cls_id)
        if cls_match and cls_match.group(1) == base_part:
          self.logger.info(f"Reconciled similar class: {predicted_class} -> {cls_id}")
          return cls_id
    
    return None

  def _parse_response(self, content: str, batch_items: List[str], classes: List[Tuple[str, str]], batch_count: int = 0) -> Tuple[List[List[float]], List[str]]:
    """
    Parse the API response to extract class probabilities.

    Args:
        content: Response content from the API
        batch_items: List of image IDs in the batch
        classes: List of (class_id, class_description) tuples
        batch_count: Current batch count (for logging)

    Returns:
        Tuple of (list of probability arrays for each image, list of corresponding batch items)
    """
    results = []
    response_texts = []
    unprocessed_items = []

    # Create a mapping from class ID to index for faster lookup
    class_id_to_idx = {cls_id: idx for idx, (cls_id, _) in enumerate(classes)}

    # Create a mapping from item ID to its position in batch_items
    item_to_idx = {item: idx for idx, item in enumerate(batch_items)}

    # First, try to extract JSON from markdown code blocks
    json_content_str = content
    
    # Check if the response is wrapped in markdown code blocks
    import re
    markdown_json_match = re.search(r'```json\s*\n(.*?)\n```', content, re.DOTALL)
    if markdown_json_match:
      json_content_str = markdown_json_match.group(1).strip()
      self.logger.debug(f"Extracted JSON from markdown code block")
    else:
      # Also try without the 'json' specifier
      markdown_match = re.search(r'```\s*\n(.*?)\n```', content, re.DOTALL)
      if markdown_match:
        potential_json = markdown_match.group(1).strip()
        # Check if it looks like JSON (starts with { and ends with })
        if potential_json.startswith('{') and potential_json.endswith('}'):
          json_content_str = potential_json
          self.logger.debug(f"Extracted JSON from generic markdown code block")

    try:
      # Try to parse as JSON
      json_content = json.loads(json_content_str)

      # Check if we have the expected structure
      if isinstance(json_content, dict):
        response_dict = {}

        # Handle different possible JSON structures
        if any(key.startswith("image_") for key in json_content.keys()):
          # Format: {"image_ID": "CLASS_ID", ...} or {"image_N": "CLASS_ID", ...}
          for key, value in json_content.items():
            if key.startswith("image_"):
              # Try to extract the item ID from the key
              item_id = key[6:]  # Remove "image_" prefix
              if item_id in item_to_idx:
                # Direct match with item ID
                response_dict[item_id] = value
              else:
                # Try to match by position (image_1, image_2, etc.)
                try:
                  idx = int(item_id) - 1
                  if 0 <= idx < len(batch_items):
                    response_dict[batch_items[idx]] = value
                except ValueError:
                  # Not a numeric index, skip
                  pass
        else:
          # Direct mapping format
          for item in batch_items:
            if item in json_content:
              response_dict[item] = json_content[item]
            elif str(item) in json_content:
              response_dict[item] = json_content[str(item)]

        # Convert to list format expected by downstream code
        response_texts = [response_dict[item] for item in batch_items if item in response_dict]
      else:
        # Handle unexpected JSON structure (like array)
        response_texts = json_content if isinstance(
          json_content, list) else [json_content]
    except json.JSONDecodeError:
      # Fall back to text parsing if JSON fails
      self.logger.warning(
        f"Failed to parse JSON response. Attempting text parsing.")
      lines = content.strip().split('\n')
      response_texts = []

      # Try to extract class IDs from text
      for item_idx, item in enumerate(batch_items):
        item_found = False
        # Look for lines that mention the item ID
        for line in lines:
          if f"Image (ID: {item})" in line or f"image_{item_idx+1}" in line.lower() or f"image {item_idx+1}" in line.lower():
            # Found a line referencing this item, now look for a class ID
            for cls_id, _ in classes:
              if cls_id in line:
                response_texts.append(cls_id)
                item_found = True
                break
            # If we found a class ID, move to the next item
            if item_found:
              break

        # If we didn't find a class ID for this item, try looking for it in the entire content
        if not item_found:
          for cls_id, _ in classes:
            # Look for patterns like "Image 1: CLASS_ID" or "image_1: CLASS_ID"
            patterns = [
              f"Image {item_idx+1}: {cls_id}",
              f"image_{item_idx+1}: {cls_id}",
              f"Image (ID: {item}): {cls_id}",
              f"{item}: {cls_id}"
            ]
            for pattern in patterns:
              if pattern in content:
                response_texts.append(cls_id)
                item_found = True
                break
            if item_found:
              break

    # Log the response for debugging
    self.logger.debug(f"Parsed response texts: {response_texts}")
    self.logger.debug(f"Batch items: {batch_items}")

    # Handle mismatch between response texts and batch items
    if len(response_texts) != len(batch_items):
      self.logger.warning(
        f"Mismatch between response texts ({len(response_texts)}) and batch items ({len(batch_items)}). Processing only valid items from batch {batch_count}.")

      # If we have fewer responses than batch items, add the unprocessed items
      if len(response_texts) < len(batch_items):
        # Items without responses are considered unprocessed
        missing_items = batch_items[len(response_texts):]
        unprocessed_items.extend(missing_items)
        self.logger.warning(f"Adding {len(missing_items)} items without responses to unprocessed list: {missing_items}")
      
      # Create a mapping between response texts and batch items
      # We'll only process items that have a valid response
      valid_items = min(len(response_texts), len(batch_items))

      # Truncate response_texts or batch_items if needed
      response_texts = response_texts[:valid_items]
      batch_items = batch_items[:valid_items]

    # Process the parsed responses
    processed_items = []
    for idx, item in enumerate(batch_items):
      if idx < len(response_texts):  # Safety check to prevent index errors
        probabilities = np.zeros(len(classes))
        append_prob = False
        
        # Get the predicted class ID
        predicted_class = response_texts[idx]
        
        # Find the index of the predicted class in the classes list
        if predicted_class in class_id_to_idx:
          cls_idx = class_id_to_idx[predicted_class]
          probabilities[cls_idx] = 1.0
          append_prob = True
          
          # Log the mapping for debugging
          self.logger.debug(
            f"Item {item}: Predicted class {predicted_class} -> Index {cls_idx}")
        else:
          # Try to find a similar class
          similar_class = self._find_similar_class(predicted_class, classes)
          if similar_class and similar_class in class_id_to_idx:
            cls_idx = class_id_to_idx[similar_class]
            probabilities[cls_idx] = 1.0
            append_prob = True
            
            # Log the reconciliation for debugging
            self.logger.debug(
              f"Item {item}: Reconciled class {predicted_class} -> {similar_class} (Index {cls_idx})")
          else:
            self.logger.warning(f"Unknown class ID: {predicted_class}")
            unprocessed_items.append(item)
          
        if append_prob:
          results.append(probabilities)
          processed_items.append(item)

    # Return both the results and the corresponding batch items, plus unprocessed items
    return results, processed_items, unprocessed_items

  def classify_images(self,
                      images: List[Tuple[str, str]],
                      classes: List[Tuple[str, str]],
                      limit: int = -1,
                      batch_size: int = 10,
                      save_frequency: int = 5) -> np.ndarray:
    """
    Classify a list of images using the GPT model.

    Args:
        images: List of (item_id, image_path) tuples
        classes: List of (class_id, class_description) tuples
        limit: Maximum number of images to process (-1 for all)
        batch_size: Number of images per batch
        save_frequency: How often to save cache (in batches)

    Returns:
        NumPy array of shape [n_images, n_classes] with class probabilities
    """
    all_probs = []
    processed_count = 0  # Track how many images we've actually processed
    all_unprocessed_items = []  # Track all unprocessed images

    self.logger.info(f"Using model: {self.model}")

    # Load system prompt
    system_prompt_path = self._get_prompt_path(self.dataset, self.test)
    with open(system_prompt_path, 'r') as file:
      system_prompt = file.read()

    # Limit images if specified
    if limit > 0:
      images = images[:limit]
      self.logger.info(f"Limiting to {limit} images")

    # Load few-shot examples if needed
    few_shot_messages = []
    if self.test in ['test_3', 'test_4']:
      few_shot_messages = self._load_few_shot_examples(self.dataset)

    batch_count = 0
    for i in tqdm(range(0, len(images), batch_size), desc="Processing Images", unit="batch"):
      # Check if we've reached the limit
      if limit > 0 and processed_count >= limit:
        self.logger.info(f"Reached limit of {limit} processed images. Stopping.")
        break
        
      batch = images[i:i + batch_size]
      batch_items = []

      # Check cache for each image in the batch
      for item, image_path in batch:
        # Skip if we've reached the limit
        if limit > 0 and processed_count >= limit:
          break
          
        cached_result = self.cache_manager.get_result(item)
        if cached_result:
          all_probs.append(cached_result)
          processed_count += 1
        else:
          batch_items.append((item, image_path))

      if not batch_items:
        continue

      batch_count += 1
      try:
        # Prepare and send API request
        messages = self._prepare_batch_request(
          batch_items, system_prompt, few_shot_messages)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        # Extract response content and token usage
        content = response.choices[0].message.content
        self.total_input_tokens += response.usage.prompt_tokens
        self.total_output_tokens += response.usage.completion_tokens

        # Save the raw response to a file for debugging
        batches_dir = os.path.join(self.base_dir, 'gpt_data', 'batches')
        os.makedirs(batches_dir, exist_ok=True)
        batch_file = os.path.join(
          batches_dir, f'{self.dataset}_{self.test}_{batch_count}.txt')
        with open(batch_file, 'w') as f:
          f.write(f"Model: {self.model}\n")
          f.write(f"Batch items: {[item for item, _ in batch_items]}\n")
          f.write(f"Response:\n{content}\n")

        # Parse response
        batch_results, processed_items, unprocessed_items = self._parse_response(
          content, [item for item, _ in batch_items], classes, batch_count)
        
        # Add unprocessed items to the global list
        all_unprocessed_items.extend(unprocessed_items)
        
        # Create the unprocessed file directory if it doesn't exist
        unprocessed_file = os.path.join(self.base_dir, 'gpt_data', 'unprocessed.txt')
        os.makedirs(os.path.dirname(unprocessed_file), exist_ok=True)
        
        # Create the unprocessed file with header if it doesn't exist
        if batch_count == 1 and not os.path.exists(unprocessed_file):
          with open(unprocessed_file, 'w') as f:
            f.write(f"# Unprocessed items file\n")
            f.write(f"# First created at {datetime.now().isoformat()}\n\n")
          self.logger.info(f"Created new unprocessed items file: {unprocessed_file}")
        
        # Write unprocessed items to file immediately if there are any
        if unprocessed_items:
          unprocessed_file = os.path.join(self.base_dir, 'gpt_data', 'unprocessed.txt')
          os.makedirs(os.path.dirname(unprocessed_file), exist_ok=True)
          with open(unprocessed_file, 'a') as f:
            for item in unprocessed_items:
              f.write(f"{item}\n")
          self.logger.info(f"Added {len(unprocessed_items)} unprocessed items to {unprocessed_file}")

        # If batch_results is empty (due to parsing error), skip this batch
        if not batch_results:
          self.logger.warning(f"No valid results from batch {batch_count}. Skipping.")
          continue

        # Create a mapping from batch items to their original paths
        item_to_path = {item: path for item, path in batch_items}

        # Add results to cache and all_probs
        for idx, item in enumerate(processed_items):
          if idx < len(batch_results):  # Safety check
            all_probs.append(batch_results[idx])
            self.cache_manager.add_result(item, batch_results[idx].tolist())
            processed_count += 1
            
            # Check if we've reached the limit after each image
            if limit > 0 and processed_count >= limit:
              self.logger.info(f"Reached limit of {limit} processed images during batch processing.")
              break

        # Periodically save cache
        self.cache_manager.save(
          periodic=True, batch_count=batch_count, save_frequency=save_frequency)

      except Exception as e:
        self.logger.error(f"Error processing batch: {e}")
        # Don't add zero arrays for failed batches - this was causing the count mismatch
        # Instead, log the error and continue

    # Final cache save
    self.cache_manager.save()

    # Calculate and display cost information
    self._display_cost_info(len(all_probs), len(images))
    
    # Log the actual number of processed images vs requested limit
    if limit > 0:
      self.logger.info(f"Requested limit: {limit}, Actual processed: {len(all_probs)}")
    
    # Log total number of unprocessed items
    if all_unprocessed_items:
      self.logger.info(f"Total unprocessed items: {len(all_unprocessed_items)}")

    return np.array(all_probs)

  def _display_cost_info(self, processed_count: int, total_count: int) -> None:
    """
    Calculate and display cost information.

    Args:
        processed_count: Number of images processed
        total_count: Total number of images
    """
    # Get cost rates for the model
    input_cost_rate, output_cost_rate = ModelConfig.get_costs(self.model)

    # Calculate total cost
    total_cost = ((self.total_input_tokens / 1_000_000) * input_cost_rate) + \
                 ((self.total_output_tokens / 1_000_000) * output_cost_rate)

    self.logger.info(f"Total input tokens used: {self.total_input_tokens}")
    self.logger.info(f"Total output tokens used: {self.total_output_tokens}")
    self.logger.info(
      f"Total cost of this call: ${total_cost:.4f}")


def encode_image(image_path: str) -> str:
  """
  Encode an image as a base64 data URL.

  Args:
      image_path: Path to the image file

  Returns:
      Base64-encoded data URL
  """
  with open(image_path, "rb") as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
  return f"data:image/jpeg;base64,{image_base64}"


def check_image_exists(item_path_tuple: Tuple[str, str]) -> Optional[Tuple[str, str]]:
  """
  Check if an image exists at the given path.
  
  Args:
      item_path_tuple: Tuple of (item_id, image_path)
      
  Returns:
      The same tuple if the image exists, None otherwise
  """
  item, image_path = item_path_tuple
  if os.path.exists(image_path):
    return (item, image_path)
  return None

def load_images_parallel(test_items: List[str], dataset_dir: str, logger=None, max_workers=None) -> List[Tuple[str, str]]:
  """
  Load images from disk in parallel.

  Args:
      test_items: List of image IDs
      dataset_dir: Directory containing the images
      logger: Logger instance
      max_workers: Number of worker processes (None = auto)

  Returns:
      List of (item_id, image_path) tuples
  """
  logger = logger or logging.getLogger("default")
  
  # Prepare the list of items and paths
  image_paths = []
  for item in test_items:
    image_path = os.path.join(dataset_dir, f"{item}.jpg")
    image_paths.append((item, image_path))
  
  # Check image existence in parallel
  with ProcessPoolExecutor(max_workers=max_workers) as executor:
    results = list(tqdm(
      executor.map(check_image_exists, image_paths),
      total=len(image_paths),
      desc="Checking images",
      unit="img"
    ))
  
  # Filter out None results (non-existent images)
  images = [result for result in results if result is not None]
  
  # Log statistics
  if logger:
    if len(images) < len(test_items):
      logger.warning(f"Failed to find {len(test_items) - len(images)} images")
  
  return images

def load_images(test_items: List[str], dataset_dir: str, logger=None) -> List[Tuple[str, str]]:
  """
  Legacy sequential image loading function.
  Kept for backward compatibility.

  Args:
      test_items: List of image IDs
      dataset_dir: Directory containing the images
      logger: Logger instance

  Returns:
      List of (item_id, image_path) tuples
  """
  logger = logger or logging.getLogger("default")
  images = []
  for item in test_items:
    image_path = os.path.join(dataset_dir, f"{item}.jpg")
    if os.path.exists(image_path):
      images.append((item, image_path))
    else:
      logger.warning(f"Image {image_path} not found.")
  return images


@click.command()
@click.option('--folders', multiple=True, default=['test_1', 'test_2'], help='List of folders to use')
@click.option('--models', multiple=True, help='List of model names to use')
@click.option('--limit', default=-1, help='Limit the number of images to process')
@click.option('--batch_size', default=10, help='Number of images per batch')
@click.option('--save_frequency', default=5, help='How often to save cache (in batches)')
@click.option('--datasets', multiple=True, default=['ArtDL'], help='List of datasets to use')
@click.option('--ignore_zero_cache', is_flag=True, help='Ignore cached results with all-zero arrays')
@click.option('--verbose', is_flag=True, help='Enable verbose logging (DEBUG level)')
def main(folders: List[str], models: List[str], limit: int, batch_size: int, save_frequency: int,
         datasets: List[str], ignore_zero_cache: bool, verbose: bool):
  """
  Main function to run the GPT image classification.

  Args:
      folders: List of test folders to use
      models: List of GPT models to use
      limit: Maximum number of images to process (-1 for all)
      batch_size: Number of images per batch
      save_frequency: How often to save cache (in batches)
      datasets: List of datasets to use
      ignore_zero_cache: Whether to ignore zero arrays in cache
      verbose: Whether to enable verbose logging
  """
  # Load OpenAI API key from config file
  script_dir = os.path.dirname(__file__)
  base_dir = os.path.join(script_dir, os.pardir)

  config = ConfigParser()
  config.read(os.path.join(script_dir, 'gpt_data', 'psw.ini'))

  api_key = config.get('openai', 'api_key', fallback=None)
  if not api_key:
    raise ValueError("OpenAI API key is not set in the config file.")

  # Process each dataset
  for dataset in datasets:
    dataset_dir = os.path.join(base_dir, 'dataset', dataset)
    dataset_data_dir = os.path.join(base_dir, 'dataset', f'{dataset}-data')

    # Load test items
    with open(os.path.join(dataset_data_dir, '2_test.txt'), 'r') as file:
      test_items = file.read().splitlines()

    # Process each test folder and model
    for folder in folders:
      for model in models:
        # Setup output folder and logger
        output_folder = os.path.join(base_dir, folder, dataset, model)
        os.makedirs(output_folder, exist_ok=True)

        # Setup logger for this specific combination
        logger = logger_utils.setup_logger(
          dataset, folder, model, output_folder, verbose)

        logger.info(
          f"Starting classification for dataset={dataset}, test={folder}, model={model}")

        # Load images using parallel processing
        images = load_images_parallel(test_items, os.path.join(
          dataset_dir, 'JPEGImages'), logger)
        logger.info(f"Number of images: {len(images)}")
        logger.info(f"Processing dataset: {dataset}")

        # Load classes
        classes_df = pd.read_csv(os.path.join(dataset_data_dir, 'classes.csv'))

        if folder in ['test_1', 'test_3']:
          classes = list(zip(classes_df['ID'], classes_df['Label']))
        elif folder in ['test_2', 'test_4']:
          classes = list(zip(classes_df['ID'], classes_df['Description']))

        logger.info(f"Processing images for test: {folder}")

        # Initialize classifier and process images
        classifier = GPTImageClassifier(
          model, api_key, dataset, folder, script_dir,
          ignore_zero_cache=ignore_zero_cache, logger=logger)

        all_probs = classifier.classify_images(
            images, classes, limit, batch_size, save_frequency
        )

        # Save results
        np.save(os.path.join(output_folder, 'probs.npy'), all_probs)
        logger.info(f"Probabilities shape: {all_probs.shape}")


if __name__ == '__main__':
  main()
