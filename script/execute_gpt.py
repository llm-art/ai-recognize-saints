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

import logger_utils
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from configparser import ConfigParser
from abc import ABC, abstractmethod
import logging
import json
import base64
import os
import click
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import openai
import re

# Increase PIL's DecompressionBombWarning threshold to ~200 million pixels
Image.MAX_IMAGE_PIXELS = 200000000

# Import custom logger


class JSONExtractor:
  """Utility class for extracting JSON content from various text formats."""

  @staticmethod
  def extract_from_markdown(content: str) -> str:
    """
    Extract JSON content from markdown code blocks.

    Args:
        content: The text content to extract JSON from

    Returns:
        Extracted JSON string or original content if no JSON found
    """
    # Check if the response is wrapped in markdown code blocks
    markdown_json_match = re.search(
      r'```json\s*\n(.*?)\n```', content, re.DOTALL)
    if markdown_json_match:
      return markdown_json_match.group(1).strip()

    # Also try without the 'json' specifier
    markdown_match = re.search(r'```\s*\n(.*?)\n```', content, re.DOTALL)
    if markdown_match:
      potential_json = markdown_match.group(1).strip()
      # Check if it looks like JSON (starts with { and ends with })
      if potential_json.startswith('{') and potential_json.endswith('}'):
        return potential_json

    return content


class ProbabilityArrayBuilder:
  """Utility class for building probability arrays from class predictions."""

  def __init__(self, num_classes: int):
    """
    Initialize the probability array builder.

    Args:
        num_classes: Number of classes in the classification problem
    """
    self.num_classes = num_classes

  def build_array(self, class_idx: int, confidence: float = 1.0) -> np.ndarray:
    """
    Build a probability array with the specified class index set to confidence.

    Args:
        class_idx: Index of the predicted class
        confidence: Confidence score (default: 1.0)

    Returns:
        NumPy array with probabilities
    """
    probabilities = np.zeros(self.num_classes)
    probabilities[class_idx] = confidence
    return probabilities


class ClassResolver:
  """Handles resolution of predicted class names to class indices."""

  def __init__(self, classes: List[Tuple[str, str]], class_adapter: 'ClassAdapter', logger=None):
    """
    Initialize the class resolver.

    Args:
        classes: List of (class_id, class_description) tuples
        class_adapter: ClassAdapter instance for alias resolution
        logger: Logger instance
    """
    self.class_id_to_idx = {cls_id: idx for idx,
                            (cls_id, _) in enumerate(classes)}
    self.classes = classes
    self.class_adapter = class_adapter
    self.logger = logger or logging.getLogger("default")

  def resolve_class(self, predicted_class: str, item: str) -> Optional[int]:
    """
    Resolve a predicted class name to a class index.

    Args:
        predicted_class: The predicted class name
        item: Item identifier for logging

    Returns:
        Class index if resolved, None otherwise
    """
    # First try direct class ID match
    if predicted_class in self.class_id_to_idx:
      cls_idx = self.class_id_to_idx[predicted_class]
      self.logger.debug(
        f"Item {item}: Direct match {predicted_class} -> Index {cls_idx}")
      return cls_idx

    # Try to find class by human-readable alias
    actual_class_id = self.class_adapter.get_class_id(predicted_class)
    if actual_class_id in self.class_id_to_idx:
      cls_idx = self.class_id_to_idx[actual_class_id]
      self.logger.debug(
        f"Item {item}: Alias {predicted_class} -> Class {actual_class_id} (Index {cls_idx})")
      return cls_idx

    # Try to find a similar class
    similar_class = self._find_similar_class(predicted_class)
    if similar_class and similar_class in self.class_id_to_idx:
      cls_idx = self.class_id_to_idx[similar_class]
      self.logger.debug(
        f"Item {item}: Reconciled {predicted_class} -> {similar_class} (Index {cls_idx})")
      return cls_idx

    self.logger.warning(f"Unknown class ID or alias: {predicted_class}")
    return None

  def _find_similar_class(self, predicted_class: str) -> Optional[str]:
    """
    Find a similar class ID if the predicted class doesn't match exactly.

    Args:
        predicted_class: The predicted class ID from the model

    Returns:
        The matching class ID if found, None otherwise
    """
    # If the predicted class is already a valid class ID, return it
    for cls_id, _ in self.classes:
      if predicted_class == cls_id:
        return cls_id

    # Check for similar class IDs (e.g., "11H(MARY)" vs "11F(MARY)")
    # Extract the base part and the description part
    match = re.match(r'([0-9]+[A-Za-z]*)(?:\(([^)]+)\))?', predicted_class)
    if match:
      base_part = match.group(1)
      desc_part = match.group(2)

      # Look for classes with the same description part
      if desc_part:
        for cls_id, _ in self.classes:
          cls_match = re.match(r'([0-9]+[A-Za-z]*)(?:\(([^)]+)\))?', cls_id)
          if cls_match and cls_match.group(2) == desc_part:
            self.logger.info(
              f"Reconciled similar class: {predicted_class} -> {cls_id}")
            return cls_id

      # Look for classes with the same base part
      for cls_id, _ in self.classes:
        cls_match = re.match(r'([0-9]+[A-Za-z]*)(?:\(([^)]+)\))?', cls_id)
        if cls_match and cls_match.group(1) == base_part:
          self.logger.info(
            f"Reconciled similar class: {predicted_class} -> {cls_id}")
          return cls_id

    return None


class JSONResponseParser:
  """Simplified parser for JSON-formatted responses only."""

  def __init__(self, logger=None):
    """
    Initialize the JSON response parser.

    Args:
        logger: Logger instance
    """
    self.logger = logger or logging.getLogger("default")

  def parse(self, content: str, batch_items: List[str]) -> Dict[str, str]:
    """
    Parse JSON response content and return ordered predictions.

    Args:
        content: Response content from the API
        batch_items: List of image IDs in the batch (in order)

    Returns:
        Dictionary mapping item IDs to predicted classes, maintaining order
    """
    # Extract JSON from markdown if present
    json_content_str = JSONExtractor.extract_from_markdown(content)

    try:
      # Parse as JSON
      json_content = json.loads(json_content_str)
      
      if not isinstance(json_content, dict):
        self.logger.error(f"Expected JSON object, got {type(json_content)}")
        return {}

      response_dict = self._parse_json_structure(json_content, batch_items)
      self.logger.debug(f"JSON parsing successful: {len(response_dict)} items parsed")
      return response_dict

    except json.JSONDecodeError as e:
      self.logger.error(f"Failed to parse JSON response: {e}")
      self.logger.debug(f"Raw content: {content}")
      return {}

  def _parse_json_structure(self, json_content: dict, batch_items: List[str]) -> Dict[str, str]:
    """
    Parse different JSON structure formats.

    Args:
        json_content: Parsed JSON content
        batch_items: List of image IDs in the batch

    Returns:
        Dictionary mapping item IDs to predicted classes
    """
    response_dict = {}

    # Handle different possible JSON structures
    if any(key.startswith("image_") for key in json_content.keys()):
      # Format: {"image_ID": "CLASS_ID", ...} or {"image_N": "CLASS_ID", ...}
      for key, value in json_content.items():
        if key.startswith("image_"):
          # Try to extract the item ID from the key
          item_id = key[6:]  # Remove "image_" prefix
          
          # Extract class from value (handle both string and dict formats)
          predicted_class = self._extract_class_from_value(value)
          
          if item_id in batch_items:
            # Direct match with item ID
            response_dict[item_id] = predicted_class
          else:
            # Try to match by position (image_1, image_2, etc.)
            try:
              idx = int(item_id) - 1
              if 0 <= idx < len(batch_items):
                response_dict[batch_items[idx]] = predicted_class
            except ValueError:
              # Not a numeric index, skip
              self.logger.warning(f"Could not parse image key: {key}")
    else:
      # Direct mapping format: {"item_id": "CLASS_ID", ...} or {"item_id": {"class": "...", "confidence": ...}, ...}
      for item in batch_items:
        if item in json_content:
          predicted_class = self._extract_class_from_value(json_content[item])
          response_dict[item] = predicted_class
        elif str(item) in json_content:
          predicted_class = self._extract_class_from_value(json_content[str(item)])
          response_dict[item] = predicted_class

    return response_dict

  def _extract_class_from_value(self, value) -> str:
    """
    Extract class name from response value, handling both string and dict formats.
    
    Args:
        value: Either a string (class name) or dict with 'class' key
        
    Returns:
        The predicted class name as string
    """
    if isinstance(value, dict):
      # Handle {"class": "CLASS_NAME", "confidence": 0.95} format
      if "class" in value:
        return str(value["class"])
      else:
        self.logger.warning(f"Dict value missing 'class' key: {value}")
        return ""
    else:
      # Handle direct string format
      return str(value)


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
      },
      "gpt-4o-2024-05-13": {
          "input_cost": 2.5,   # Cost per 1M input tokens
          "output_cost": 10.0  # Cost per 1M output tokens
      },
      "gpt-4o-2024-08-06": {
          "input_cost": 2.5,   # Cost per 1M input tokens
          "output_cost": 10.0  # Cost per 1M output tokens
      },
      "gpt-4o-mini-2024-07-18": {
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

  def __init__(self, base_dir: str, dataset: str, test: str, model: str, logger=None):
    """
    Initialize the cache manager.

    Args:
        base_dir: Base directory for the project
        dataset: Dataset name (e.g., 'ArtDL')
        test: Test identifier (e.g., 'test_1')
        model: Model name (e.g., 'gpt-4o')
        logger: Logger instance
    """
    # Use the same directory structure as probability files: {test}/{dataset}/{model}
    self.cache_dir = os.path.join(base_dir, os.pardir, test, dataset, model)
    os.makedirs(self.cache_dir, exist_ok=True)

    # Store cache file in the same directory as probability files
    self.cache_file = os.path.join(self.cache_dir, 'cache.json')

    self.metadata = {
        "model": model,
        "dataset": dataset,
        "test": test,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0"
    }

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
    return self.cache.get(image_id)

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


class ClassAdapter:
  """Adapter to convert between complex class IDs and human-readable aliases."""

  def __init__(self, classes: List[Tuple[str, str]]):
    """
    Initialize the class adapter.

    Args:
        classes: List of (class_id, class_label) tuples
    """
    self.id_to_alias = {}
    self.alias_to_id = {}

    for class_id, class_label in classes:
      # Create human-readable alias from class_id
      alias = self._create_alias(class_id)
      self.id_to_alias[class_id] = alias
      self.alias_to_id[alias] = class_id

  def _create_alias(self, class_id: str) -> str:
    """
    Create a human-readable alias from a class ID by extracting the saint's name.

    Args:
        class_id: The class ID (e.g., '11H(PAUL)', '11HH(MARY MAGDALENE)')

    Returns:
        Human-readable alias (e.g., 'paul', 'mary_magdalene')
    """
    import re
    # Extract content within parentheses
    match = re.search(r'\(([^)]+)\)', class_id)
    if match:
      name = match.group(1)
      # Convert to lowercase and replace spaces with underscores
      alias = name.lower().replace(' ', '_').replace('.', '').replace(',', '')
      # Remove common prefixes
      alias = alias.replace('st_', '').replace(
        'saint_', '').replace('the_', '')
      return alias

    # Fallback to original logic if no parentheses found
    alias = class_id.lower().replace(' ', '_').replace('.', '').replace(',', '')
    alias = alias.replace('st_', '').replace('saint_', '').replace('the_', '')
    return alias

  def get_alias(self, class_id: str) -> str:
    """Get human-readable alias for a class ID."""
    return self.id_to_alias.get(class_id, class_id)

  def get_class_id(self, alias: str) -> str:
    """Get class ID from human-readable alias."""
    return self.alias_to_id.get(alias, alias)

  def get_all_aliases(self) -> List[Tuple[str, str]]:
    """Get all (alias, class_id) pairs."""
    return [(alias, class_id) for class_id, alias in self.id_to_alias.items()]


class GPTImageClassifier:
  """Classifies images using OpenAI's GPT models with vision capabilities."""

  def __init__(self, model: str, api_key: str, dataset: str, test: str, base_dir: str,
               temperature: float = 0.0, top_p: float = 0.1, seed: int = 12345, logger=None):
    """
    Initialize the classifier.

    Args:
        model: The GPT model to use (e.g., 'gpt-4o')
        api_key: OpenAI API key
        dataset: Dataset name
        test: Test identifier
        base_dir: Base directory for the project
        temperature: Temperature for generation
        top_p: Top-p (nucleus sampling) for generation
        seed: Seed for deterministic results
        logger: Logger instance
    """
    self.model = model
    self.client = openai.Client(api_key=api_key)
    self.dataset = dataset
    self.test = test
    self.base_dir = base_dir

    # Store hyperparameters
    self.temperature = temperature
    self.top_p = top_p
    self.seed = seed

    # Set up logger
    self.logger = logger or logging.getLogger("default")

    # Initialize cache manager
    self.cache_manager = CacheManager(
      base_dir, dataset, test, model, logger=self.logger)

    # Initialize prompt folder
    self.prompt_folder = os.path.join(base_dir, os.pardir, 'prompts')

    # Token usage tracking
    self.total_input_tokens = 0
    self.total_output_tokens = 0

    # Initialize class adapter (will be set during prompt generation)
    self.class_adapter = None

  def _generate_prompt(self, dataset: str, test: str, classes: List[Tuple[str, str]]) -> str:
    """
    Generate the system prompt by combining base template with dynamic class list.

    Args:
        dataset: Dataset name
        test: Test identifier
        classes: List of (class_id, class_description) tuples

    Returns:
        Complete system prompt string
    """
    # Load base prompt template
    base_template_path = os.path.join(
      self.prompt_folder, 'base_prompt_template.txt')
    if not os.path.exists(base_template_path):
      raise FileNotFoundError(
        f"Base prompt template not found: {base_template_path}")

    with open(base_template_path, 'r') as f:
      base_template = f.read()

    # Create class adapter for human-readable aliases
    self.class_adapter = ClassAdapter(classes)

    # Generate few-shot examples section for test_3
    if test == 'test_3':
      few_shot_section = self._generate_few_shot_section(dataset)
    else:
      few_shot_section = ""

    # Generate class list using human-readable aliases
    class_list_lines = []
    class_list_lines.append(
      "Each <CATEGORY_ID> must be one of (use only the category ID as output):")
    class_list_lines.append("")

    for class_id, class_desc in classes:
      alias = self.class_adapter.get_alias(class_id)
      class_list_lines.append(f'"{alias}" - {class_desc}')

    class_list = "\n".join(class_list_lines)

    # Replace placeholders with generated content
    complete_prompt = base_template.replace(
      "{FEW_SHOT_EXAMPLES}", few_shot_section)
    complete_prompt = complete_prompt.replace("{CLASS_LIST}", class_list)

    # Log the generated prompt
    self.logger.info("=== GENERATED PROMPT ===")
    self.logger.info(f"Dataset: {dataset}, Test: {test}")
    self.logger.info(
      f"Hyperparameters: temperature={self.temperature}, top_p={self.top_p}, seed={self.seed}")
    self.logger.info("Prompt content:")
    self.logger.info(complete_prompt)
    self.logger.info("=== END PROMPT ===")

    return complete_prompt

  def _generate_few_shot_section(self, dataset: str) -> str:
    """
    Generate the few-shot examples section for test_3 prompts.

    Args:
        dataset: Dataset name

    Returns:
        Formatted few-shot section string
    """
    few_shot_folder = os.path.join(
      self.base_dir, os.pardir, 'dataset', f"{dataset}-data", 'few-shot')
    few_shot_file = os.path.join(few_shot_folder, 'train_data.csv')

    if not os.path.exists(few_shot_file):
      self.logger.warning(f"Few-shot file not found: {few_shot_file}")
      return ""

    try:
      few_shot_df = pd.read_csv(few_shot_file)

      if few_shot_df.empty:
        self.logger.warning(f"Few-shot file is empty: {few_shot_file}")
        return ""

      # Generate the few-shot examples list
      examples_list = []
      for _, row in few_shot_df.iterrows():
        # Use aliases for the class IDs in the few-shot section
        class_id = row['class']
        alias = self.class_adapter.get_alias(class_id)
        examples_list.append(f'  "{row["item"]}", "{alias}"')

      few_shot_section = f"""You will be shown {len(few_shot_df)} example images categorized as follows:
{{
{chr(10).join(examples_list)}
}}

"""
      return few_shot_section

    except Exception as e:
      self.logger.error(f"Error generating few-shot section: {e}")
      return ""

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

  def _parse_response(self, content: str, batch_items: List[str], classes: List[Tuple[str, str]], batch_count: int = 0) -> Dict[str, np.ndarray]:
    """
    Parse the API response to extract class probabilities with positional tracking.

    Args:
        content: Response content from the API
        batch_items: List of image IDs in the batch (in order)
        classes: List of (class_id, class_description) tuples
        batch_count: Current batch count (for logging)

    Returns:
        Dictionary mapping item IDs to probability arrays
    """
    # Initialize parsing components
    response_parser = JSONResponseParser(self.logger)
    class_resolver = ClassResolver(classes, self.class_adapter, self.logger)
    prob_builder = ProbabilityArrayBuilder(len(classes))

    # Parse the response content to extract predicted classes
    response_dict = response_parser.parse(content, batch_items)

    # Log the response for debugging
    self.logger.debug(f"Parsed response dict: {response_dict}")
    self.logger.debug(f"Batch items: {batch_items}")

    # Process predictions for each item in the batch
    results = {}
    
    for item in batch_items:
      if item in response_dict:
        predicted_class = response_dict[item]
        
        # Resolve the predicted class to a class index
        cls_idx = class_resolver.resolve_class(predicted_class, item)
        
        if cls_idx is not None:
          # Build probability array
          probabilities = prob_builder.build_array(cls_idx, 1.0)
          results[item] = probabilities
          
          # Log the successful resolution
          self.logger.debug(f"Item {item}: Successfully resolved to index {cls_idx}")
        else:
          # Could not resolve the class - create uniform distribution
          uniform_probs = np.full(len(classes), 1.0 / len(classes))
          results[item] = uniform_probs
          self.logger.warning(f"Item {item}: Could not resolve class '{predicted_class}', using uniform distribution")
      else:
        # No prediction for this item - create uniform distribution
        uniform_probs = np.full(len(classes), 1.0 / len(classes))
        results[item] = uniform_probs
        self.logger.warning(f"Item {item}: No prediction in response, using uniform distribution")

    return results

  def classify_images(self,
                      images: List[Tuple[str, str]],
                      classes: List[Tuple[str, str]],
                      limit: int = -1,
                      batch_size: int = 10,
                      save_frequency: int = 5) -> np.ndarray:
    """
    Classify a list of images using the GPT model with positional tracking.

    Args:
        images: List of (item_id, image_path) tuples
        classes: List of (class_id, class_description) tuples
        limit: Maximum number of images to process (-1 for all)
        batch_size: Number of images per batch
        save_frequency: How often to save cache (in batches)

    Returns:
        NumPy array of shape [n_images, n_classes] with class probabilities
    """
    self.logger.info(f"Using model: {self.model}")

    # Generate system prompt dynamically
    system_prompt = self._generate_prompt(self.dataset, self.test, classes)

    # Limit images if specified
    if limit > 0:
      images = images[:limit]
      self.logger.info(f"Limiting to {limit} images")

    # Pre-allocate results array with correct size to maintain positional alignment
    total_images = len(images)
    all_probs = [None] * total_images
    
    # Track failed items for potential repair
    failed_items = {}  # {position: (item_id, image_path)}
    
    self.logger.info(f"Pre-allocated results array for {total_images} images")

    # Load few-shot examples if needed
    few_shot_messages = []
    if self.test in ['test_3', 'test_4']:
      few_shot_messages = self._load_few_shot_examples(self.dataset)

    batch_count = 0
    for i in tqdm(range(0, len(images), batch_size), desc="Processing Images", unit="batch"):
      batch = images[i:i + batch_size]
      batch_items = []
      batch_positions = []  # Track original positions in the full image list

      # Check cache for each image in the batch
      for idx, (item, image_path) in enumerate(batch):
        original_pos = i + idx
        
        cached_result = self.cache_manager.get_result(item)
        if cached_result:
          all_probs[original_pos] = np.array(cached_result)
          self.logger.debug(f"Using cached result for {item} at position {original_pos}")
        else:
          batch_items.append((item, image_path))
          batch_positions.append(original_pos)

      if not batch_items:
        continue

      batch_count += 1
      try:
        # Prepare and send API request
        messages = self._prepare_batch_request(
          batch_items, system_prompt, few_shot_messages)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            response_format={"type": "json_object"}
        )

        # Extract response content and token usage
        content = response.choices[0].message.content
        self.total_input_tokens += response.usage.prompt_tokens
        self.total_output_tokens += response.usage.completion_tokens

        # Log system fingerprint for determinism tracking
        system_fingerprint = getattr(response, 'system_fingerprint', 'N/A')
        self.logger.info(
          f"Batch {batch_count}: system_fingerprint={system_fingerprint}")

        # Save the raw response to a file for debugging in batches/ subfolder
        batches_dir = os.path.join(
          self.base_dir, os.pardir, self.test, self.dataset, self.model, 'batches')
        os.makedirs(batches_dir, exist_ok=True)
        batch_file = os.path.join(batches_dir, f'{batch_count}.txt')
        with open(batch_file, 'w') as f:
          f.write(f"Model: {self.model}\n")
          f.write(f"System fingerprint: {system_fingerprint}\n")
          f.write(f"Batch items: {[item for item, _ in batch_items]}\n")
          f.write(f"Batch positions: {batch_positions}\n")
          f.write(f"Response:\n{content}\n")

        # Parse response - returns dict mapping item_id -> probability array
        batch_results = self._parse_response(
          content, [item for item, _ in batch_items], classes, batch_count)

        # Insert results at correct positions
        for idx, (item, image_path) in enumerate(batch_items):
          original_pos = batch_positions[idx]
          
          if item in batch_results:
            # Successfully processed - insert at correct position
            all_probs[original_pos] = batch_results[item]
            self.cache_manager.add_result(item, batch_results[item].tolist())
            self.logger.debug(f"Inserted result for {item} at position {original_pos}")
          else:
            # Failed to process - track for potential repair
            failed_items[original_pos] = (item, image_path)
            self.logger.warning(f"Failed to process {item} at position {original_pos}")

        # Periodically save cache
        self.cache_manager.save(
          periodic=True, batch_count=batch_count, save_frequency=save_frequency)

      except Exception as e:
        self.logger.error(f"Error processing batch {batch_count}: {e}")
        
        # Track all items in this batch as failed
        for idx, (item, image_path) in enumerate(batch_items):
          original_pos = batch_positions[idx]
          failed_items[original_pos] = (item, image_path)

    # Fill any remaining None positions with uniform distribution
    uniform_probs = np.full(len(classes), 1.0 / len(classes))
    failed_count = 0
    
    for i, prob in enumerate(all_probs):
      if prob is None:
        all_probs[i] = uniform_probs
        failed_count += 1

    # Save failed items for potential repair
    if failed_items:
      failed_file = os.path.join(
        self.base_dir, os.pardir, self.test, self.dataset, self.model, 'failed_items.json')
      os.makedirs(os.path.dirname(failed_file), exist_ok=True)
      
      with open(failed_file, 'w') as f:
        json.dump(failed_items, f, indent=2)
      
      self.logger.info(f"Saved {len(failed_items)} failed items to {failed_file}")

    # Final cache save
    self.cache_manager.save()

    # Calculate and display cost information
    self._display_cost_info(total_images - failed_count, total_images)

    # Log statistics
    self.logger.info(f"Total images processed: {total_images}")
    self.logger.info(f"Successfully processed: {total_images - failed_count}")
    self.logger.info(f"Failed/uniform placeholders: {failed_count}")
    
    if failed_count > 0:
      self.logger.warning(f"Used uniform distribution for {failed_count} failed predictions")

    # Convert to numpy array - all positions are now guaranteed to be filled
    result_array = np.array(all_probs)
    self.logger.info(f"Final probabilities shape: {result_array.shape}")
    
    return result_array

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


@click.command()
@click.option('--folders', multiple=True, help='List of folders to use')
@click.option('--models', multiple=True, help='List of model names to use')
@click.option('--limit', default=-1, help='Limit the number of images to process')
@click.option('--batch_size', default=1, help='Number of images per batch')
@click.option('--save_frequency', default=5, help='How often to save cache (in batches)')
@click.option('--datasets', multiple=True, help='List of datasets to use')
@click.option('--verbose', is_flag=True, help='Enable verbose logging (DEBUG level)')
@click.option('--temperature', default=0.0, help='Temperature for generation (default: 0.0, min: 0.0)')
@click.option('--top_p', default=0.1, help='Top-p (nucleus sampling) for generation (default: 0.1)')
@click.option('--seed', default=12345, help='Seed for deterministic results (default: 12345)')
@click.option('--clean', is_flag=True, help='Remove cache and logs from previous runs before starting')
def main(folders: List[str], models: List[str], limit: int, batch_size: int, save_frequency: int,
         datasets: List[str], verbose: bool, temperature: float, top_p: float, seed: int, clean: bool):
  """
  Main function to run the GPT image classification.

  Args:
      folders: List of test folders to use
      models: List of GPT models to use
      limit: Maximum number of images to process (-1 for all)
      batch_size: Number of images per batch
      save_frequency: How often to save cache (in batches)
      datasets: List of datasets to use
      verbose: Whether to enable verbose logging
      clean: Whether to remove cache and logs from previous runs
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
        # Setup output folder
        output_folder = os.path.join(base_dir, folder, dataset, model)
        os.makedirs(output_folder, exist_ok=True)

        # Clean previous runs if requested (BEFORE setting up logger)
        if clean:
          import shutil
          # Remove cache and logs from the model directory
          model_dir = os.path.join(base_dir, folder, dataset, model)
          if os.path.exists(model_dir):
            # List of files/directories to remove
            items_to_remove = []

            # Cache file
            cache_file = os.path.join(model_dir, 'cache.json')
            if os.path.exists(cache_file):
              items_to_remove.append(cache_file)

            # Batches directory
            batches_dir = os.path.join(model_dir, 'batches')
            if os.path.exists(batches_dir):
              items_to_remove.append(batches_dir)

            # Log files - check multiple possible log file patterns
            log_patterns = [
              f'{dataset}_{folder}_{model}.log',  # Original pattern
              f'{model}.log',                     # Simple model name pattern
              f'{folder}_{model}.log',            # Folder + model pattern
            ]
            
            for log_pattern in log_patterns:
              log_file = os.path.join(model_dir, log_pattern)
              if os.path.exists(log_file):
                items_to_remove.append(log_file)

            # Unprocessed items file
            unprocessed_file = os.path.join(model_dir, 'unprocessed.txt')
            if os.path.exists(unprocessed_file):
              items_to_remove.append(unprocessed_file)

            # Failed items file
            failed_file = os.path.join(model_dir, 'failed_items.json')
            if os.path.exists(failed_file):
              items_to_remove.append(failed_file)

            # Remove the items
            for item in items_to_remove:
              try:
                if os.path.isfile(item):
                  os.remove(item)
                  print(f"Removed file: {item}")
                elif os.path.isdir(item):
                  shutil.rmtree(item)
                  print(f"Removed directory: {item}")
              except Exception as e:
                print(f"Warning: Could not remove {item}: {e}")

            print(f"Cleaned previous cache and logs for dataset={dataset}, test={folder}, model={model}")

        # Setup logger AFTER cleaning (so it can create a fresh log file)
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
          temperature=temperature, top_p=top_p, seed=seed, logger=logger)

        all_probs = classifier.classify_images(
            images, classes, limit, batch_size, save_frequency
        )

        # Save results
        np.save(os.path.join(output_folder, 'probs.npy'), all_probs)
        logger.info(f"Probabilities shape: {all_probs.shape}")


if __name__ == '__main__':
  main()
