# sam2/build_sam.py

import logging
import os

import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

def build_sam2_video_predictor(
    config_path,
    config_name,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
):
    """
    Initialize and return the SAM2 Video Predictor model.

    Args:
        config_path (str): Relative path to the configuration directory.
        config_name (str): Name of the configuration file without the .yaml extension.
        ckpt_path (str, optional): Relative path to the model checkpoint file. Defaults to None.
        device (str, optional): Device to load the model on ('cuda' or 'cpu'). Defaults to "cuda".
        mode (str, optional): Mode to set the model ('eval' or 'train'). Defaults to "eval".
        hydra_overrides_extra (list, optional): Additional Hydra overrides. Defaults to [].
        apply_postprocessing (bool, optional): Whether to apply postprocessing steps. Defaults to True.

    Returns:
        SAM2VideoPredictor: Initialized SAM2 video predictor model.
    """
    # Clear Hydra if it's already initialized to allow re-initialization
    if GlobalHydra.instance().is_initialized():
        logging.info("Clearing existing Hydra instance.")
        GlobalHydra.instance().clear()

    # Base Hydra overrides for the video predictor
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # Dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # Sigmoid mask logits on interacted frames with clicks in the memory encoder
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # Fill small holes in the low-res masks up to `fill_hole_area` before resizing to original video resolution
            "++model.fill_hole_area=8",
        ]
        hydra_overrides.extend(hydra_overrides_extra)

    try:
        # Initialize Hydra with the specified config path
        with initialize(config_path=config_path, job_name="build_sam2_video_predictor"):
            # Compose the configuration using the config name and any overrides
            cfg = compose(config_name=config_name, overrides=hydra_overrides)
    except Exception as e:
        logging.error(f"Error during Hydra initialization: {e}")
        raise RuntimeError(f"Hydra initialization failed: {e}")

    # Resolve any interpolations in the config
    OmegaConf.resolve(cfg)

    # Instantiate the model from the configuration
    try:
        model = instantiate(cfg.model, _recursive_=True)
    except Exception as e:
        logging.error(f"Error instantiating the model: {e}")
        raise RuntimeError(f"Model instantiation failed: {e}")

    # Load the checkpoint if provided
    if ckpt_path is not None:
        try:
            _load_checkpoint(model, ckpt_path)
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}")
            raise RuntimeError(f"Checkpoint loading failed: {e}")

    # Move the model to the specified device
    model = model.to(device)

    # Set the model to evaluation mode if specified
    if mode == "eval":
        model.eval()

    return model

def _load_checkpoint(model, ckpt_path):
    """
    Load the model checkpoint.

    Args:
        model (torch.nn.Module): The model to load the checkpoint into.
        ckpt_path (str): Relative path to the checkpoint file.

    Raises:
        RuntimeError: If there are missing or unexpected keys in the state_dict.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}")

    # Load the checkpoint (state_dict) from the specified path
    try:
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
    except KeyError:
        raise KeyError(f"The checkpoint file {ckpt_path} does not contain a 'model' key.")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    # Load the state_dict into the model with strict=False to allow partial loading
    missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)

    # Handle missing keys
    if missing_keys:
        logging.warning(f"Missing keys in state_dict: {missing_keys}")

    # Handle unexpected keys
    if unexpected_keys:
        logging.warning(f"Unexpected keys in state_dict: {unexpected_keys}")

    if missing_keys or unexpected_keys:
        logging.warning("Checkpoint loaded with missing or unexpected keys.")
    else:
        logging.info("Checkpoint loaded successfully.")
