#!/usr/bin/env python3
"""
Logging Utilities for Image Classification Scripts

This module provides a consistent logging setup for image classification scripts,
ensuring logs are properly formatted and don't interfere with tqdm progress bars.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from tqdm import tqdm

class TqdmLoggingHandler(logging.Handler):
    """Custom logging handler that writes to console without disrupting tqdm progress bars"""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_logger(dataset, test, model, output_folder, verbose=False):
    """
    Set up a logger for a specific dataset/test/model combination.
    
    Args:
        dataset: Dataset name (e.g., 'ArtDL')
        test: Test identifier (e.g., 'test_1')
        model: Model name (e.g., 'gpt-4o')
        output_folder: Directory where results and logs will be saved
        verbose: Whether to enable DEBUG level logging
        
    Returns:
        A configured logger instance
    """
    # Create a unique logger name for this combination
    logger_name = f"output.log"
    logger = logging.getLogger(logger_name)
    
    # Set log level based on verbose flag
    log_level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(log_level)
    
    # Clear any existing handlers (in case logger already exists)
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # File handler (writes to log file in the output folder)
    log_file = os.path.join(output_folder, f"{model}.log")
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    
    # Console handler that doesn't interfere with tqdm
    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logger initialized for {dataset}/{test}/{model}")
    return logger
