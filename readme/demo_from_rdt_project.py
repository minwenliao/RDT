# Please first clone the repository and install dependencies
# Then switch to the root directory of the repository by "cd RoboticsDiffusionTransformer"

# Import a create function from thae code base
from scripts.agilex_model import create_model
import torch

# Names of cameras used for visual input
#CAMERA_NAMES = ['cam_high', 'cam_right_wrist', 'cam_left_wrist']
CAMERA_NAMES = ['camera_f', 'camera_l', 'camera_r']
config = {
    'episode_len': 1000,  # Max length of one episode
    'state_dim': 14,      # Dimension of the robot's state
    'chunk_size': 64,     # Number of actions to predict in one step
    'camera_names': CAMERA_NAMES,
}
#pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384" 
pretrained_vision_encoder_name_or_path = "/home/agilex/RoboticsDiffusionTransformer-main/google/siglip-so400m-patch14-384"

# Create the model with the specified configuration
model = create_model(
    args=config,
    dtype=torch.bfloat16, 
    pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
    pretrained="/media/agilex/8447-7ADF/rdt-1b",
    control_frequency=25,
)

# Start inference process
# Load the pre-computed language embeddings
# Refer to scripts/encode_lang.py for how to encode the language instruction
lang_embeddings_path = 'outs/handover_pan.pt'
text_embedding = torch.load(lang_embeddings_path)['embeddings']  
images: List(PIL.Image) = ... #  The images from last 2 frames
proprio = ... # The current robot state
# Perform inference to predict the next `chunk_size` actions
actions = policy.step(
    proprio=proprio,
    images=images,
    text_embeds=text_embedding 
)
