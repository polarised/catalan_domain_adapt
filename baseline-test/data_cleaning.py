from data_utils import data_loader, save_pickle
import os

conll_file_location = r'C:\Users\mikel\OneDrive\Desktop\Uni\4th Year\1st Semester\Adv. Automatic Learning\Project\Code\Preprocessing\Datasets\ca_ancora-ud-test.conllu'

save_dir = r'C:\Users\mikel\OneDrive\Desktop\Uni\4th Year\1st Semester\Adv. Automatic Learning\Project\Code\Preprocessing\Clean_Datasets'
save_path = os.path.join(save_dir, "cleaned_ud_dataset(cat)_completo.pkl")

# Ensure directory exists
os.makedirs(save_dir, exist_ok=True)

loaded_dataset = data_loader(conll_file_location)

save_pickle(loaded_dataset, save_path)