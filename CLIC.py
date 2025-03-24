import os
import csv
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.ops.boxes import masks_to_boxes
from torch import nn, optim
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import utils
from engine import train_one_epoch, evaluate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from concurrent.futures import ProcessPoolExecutor, as_completed
from torchvision.transforms import v2 as T
# ============================
# Utilisation du GPU si dispo
# ============================
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


############################################################
# (A) EXTRACTION & SAUVEGARDE DES TRANCHES POUR TOUTES LES VUES - OPTIMISEE
############################################################

def process_volume(scan_file, mask_file, scans_folder, masks_folder, temp_folder, min_box_size):
    """
    Traite un volume (scan et masque) pour extraire les slices valides
    dans les trois vues (axial, coronal, sagittal). En plus des slices positives,
    on conserve un échantillon de slices négatives (sans aucun voxel non nul dans le masque).
    Renvoie une liste de lignes pour le CSV sous la forme [slice_id, view, scan_npy, mask_npy].
    """
    results = []
    scan_path = os.path.join(scans_folder, scan_file)
    mask_path = os.path.join(masks_folder, mask_file)
    
    try:
        # Charger les volumes en mémoire via mmap
        scan_img = nib.load(scan_path, mmap=True)
        mask_img = nib.load(mask_path, mmap=True)
        # Récupérer les données sous forme de tableaux NumPy
        scan_data = np.asanyarray(scan_img.dataobj)
        mask_data = np.asanyarray(mask_img.dataobj)
        mask_data = np.round(mask_data).astype(np.int32)
        vol_shape = scan_img.shape  # (X, Y, Z)
    except Exception as e:
        print(f"Erreur chargement {scan_file} ou {mask_file}: {e}")
        return results

    # Dictionnaire des vues et leur axe correspondant
    views = {
        'axial': 2,    # slices le long de l'axe Z
        'coronal': 1,  # slices le long de l'axe Y
        'sagittal': 0  # slices le long de l'axe X
    }
    
    # Déterminer le nombre maximum de slices positives à extraire pour ce volume
    max_slices = 40 if scan_file.startswith("MN") else 90

    # Parcours des vues
    for view, axis in views.items():
        num_slices = vol_shape[axis]
        valid_slices = []
        negative_slices = []
        
        # Itérer sur toutes les slices de la vue
        for i in range(num_slices):
            if view == 'axial':
                scan_slice = scan_data[..., i]
                mask_slice = mask_data[..., i]
            elif view == 'coronal':
                scan_slice = scan_data[:, i, :]
                mask_slice = mask_data[:, i, :]
            elif view == 'sagittal':
                scan_slice = scan_data[i, ...]
                mask_slice = mask_data[i, ...]
            
            # Vérifier si la slice contient des annotations
            non_zero = np.nonzero(mask_slice)
            if non_zero[0].size == 0:
                negative_slices.append(i)
                continue  # C'est une slice négative
            # Calcul de la bounding box
            y_min, y_max = np.min(non_zero[0]), np.max(non_zero[0])
            x_min, x_max = np.min(non_zero[1]), np.max(non_zero[1])
            height = y_max - y_min
            width = x_max - x_min
            if height < min_box_size or width < min_box_size:
                continue
            num_non_zero = np.sum(mask_slice != 0)
            valid_slices.append((i, num_non_zero))

        valid_slices.sort(key=lambda x: x[1], reverse=True)
        selected_positive = [idx for idx, _ in valid_slices[:max_slices]]
        
        # Sauvegarder les slices positives
        for i in selected_positive:
            if view == 'axial':
                scan_slice = scan_data[..., i]
                mask_slice = mask_data[..., i]
            elif view == 'coronal':
                scan_slice = scan_data[:, i, :]
                mask_slice = mask_data[:, i, :]
            elif view == 'sagittal':
                scan_slice = scan_data[i, ...]
                mask_slice = mask_data[i, ...]
            
            min_val, max_val = scan_slice.min(), scan_slice.max()
            if (max_val - min_val) > 1e-8:
                scan_slice_norm = (scan_slice - min_val) / (max_val - min_val)
            else:
                scan_slice_norm = np.zeros_like(scan_slice)
            
            base_id = f"{os.path.splitext(scan_file)[0]}_{view}_{i:04d}"
            scan_outfile = os.path.join(temp_folder, f"scan_{base_id}.npy")
            mask_outfile = os.path.join(temp_folder, f"mask_{base_id}.npy")
            np.save(scan_outfile, scan_slice_norm)
            np.save(mask_outfile, mask_slice)
            results.append([base_id, view, scan_outfile, mask_outfile])
        

        nb_negatives = 15
        if len(negative_slices) > 0:

            chosen_negatives = np.random.choice(negative_slices, size=min(nb_negatives, len(negative_slices)), replace=False)
            for i in chosen_negatives:
                if view == 'axial':
                    scan_slice = scan_data[..., i]
                elif view == 'coronal':
                    scan_slice = scan_data[:, i, :]
                elif view == 'sagittal':
                    scan_slice = scan_data[i, ...]
                # Normalisation
                min_val, max_val = scan_slice.min(), scan_slice.max()
                if (max_val - min_val) > 1e-8:
                    scan_slice_norm = (scan_slice - min_val) / (max_val - min_val)
                else:
                    scan_slice_norm = np.zeros_like(scan_slice)
                # Créer un masque vide (négatif)
                mask_slice = np.zeros_like(scan_slice, dtype=np.int32)
                base_id = f"{os.path.splitext(scan_file)[0]}_{view}_NEG_{i:04d}"
                scan_outfile = os.path.join(temp_folder, f"scan_{base_id}.npy")
                mask_outfile = os.path.join(temp_folder, f"mask_{base_id}.npy")
                np.save(scan_outfile, scan_slice_norm)
                np.save(mask_outfile, mask_slice)
                results.append([base_id, view, scan_outfile, mask_outfile])
                
    return results

def extract_and_save_slices_optimized(scans_folder, masks_folder, temp_folder, csv_path, min_box_size=35, num_workers=4):
    """
    Version optimisée et parallélisée de l'extraction.
    Parcourt les volumes de scans et masques, traite chaque volume en parallèle
    et écrit le CSV contenant les slices (positives et négatives) issues de toutes les vues.
    """
    os.makedirs(temp_folder, exist_ok=True)
    scan_files = sorted([f for f in os.listdir(scans_folder) if f.endswith('.nii')])
    mask_files = sorted([f for f in os.listdir(masks_folder) if f.endswith('.nii')])
    assert len(scan_files) == len(mask_files), "Mismatch entre scans et masques!"
    
    all_rows = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for scan_file, mask_file in zip(scan_files, mask_files):
            futures.append(executor.submit(process_volume, scan_file, mask_file,
                                             scans_folder, masks_folder, temp_folder, min_box_size))
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing volumes"):
            result = future.result()
            all_rows.extend(result)
    
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['slice_id', 'view', 'scan_npy', 'mask_npy'])
        writer.writerows(all_rows)
    
    print(f"Extraction terminée. Les slices validées sont dans {temp_folder}")
    print(f"Métadonnées enregistrées dans : {csv_path}")


############################################################
# (B) DATASET
############################################################

class TempSlicesDataset(Dataset):
    def __init__(self, csv_path, transforms=None):
        """
        On suppose que le CSV contient des slices positives et négatives.
        Les colonnes du CSV sont: slice_id, view, scan_npy, mask_npy.
        """
        self.transforms = transforms
        self.entries = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                slice_id = row['slice_id']  # identifiant unique
                view = row['view']          # vue (axial, coronal, sagittal)
                scan_npy = row['scan_npy']
                mask_npy = row['mask_npy']
                self.entries.append((slice_id, view, scan_npy, mask_npy))
    
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        slice_id, view, scan_path, mask_path = self.entries[idx]
        scan_slice = np.load(scan_path)  # [H, W]
        mask_slice = np.load(mask_path)  # [H, W]
        scan_tensor = torch.from_numpy(scan_slice).unsqueeze(0).float()  # [1, H, W]
        mask_tensor = torch.from_numpy(mask_slice).long()                # [H, W]
        
        # Extraction des labels (> 0) présents dans le masque
        labels = torch.unique(mask_tensor)
        labels = labels[labels != 0]
        num_classes = 4  # À ajuster si besoin
        
        num_objs = len(labels)
        H, W = mask_tensor.shape
        masks = torch.zeros((num_objs, H, W), dtype=torch.uint8)
        for i, lbl in enumerate(labels):
            masks[i] = (mask_tensor == lbl).byte()
        
        # Calcul des bounding boxes
        boxes = masks_to_boxes(masks)  # shape [num_objs, 4]
        valid_boxes, valid_labels, valid_masks = [], [], []
        for i, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = box
            if (x_max > x_min) and (y_max > y_min) and labels[i] < 5:
                valid_boxes.append(box)
                valid_labels.append(labels[i])
                valid_masks.append(masks[i])
        
        if len(valid_boxes) == 0:
            # Cas négatif : aucune annotation
            return scan_tensor, {
                "boxes": tv_tensors.BoundingBoxes(torch.empty((0, 4), dtype=torch.float32), format="XYXY", canvas_size=F.get_size(scan_tensor)),
                "masks": tv_tensors.Mask(torch.empty((0, mask_tensor.shape[0], mask_tensor.shape[1]), dtype=torch.uint8)),
                "labels": torch.empty((0,), dtype=torch.int64),
                "image_id": idx,
                "area": torch.empty((0,), dtype=torch.float32),
                "iscrowd": torch.empty((0,), dtype=torch.int64)
            }
        
        boxes = torch.stack(valid_boxes)
        labels = torch.tensor(valid_labels, dtype=torch.int64)
        masks = torch.stack(valid_masks)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        if self.transforms is not None:
            scan_tensor = self.transforms(scan_tensor)
        
        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(scan_tensor)),
            "masks": tv_tensors.Mask(masks),
            "labels": labels,
            "image_id": idx,
            "area": area,
            "iscrowd": iscrowd
        }
        return scan_tensor, target


############################################################
# (C) COLLATE_FN (optionnel)
############################################################

def collate_remove_none(batch):
    batch = [b for b in batch if b is not None]
    return utils.collate_fn(batch)


############################################################
# (D) MODÈLE + MÉTRIQUES
############################################################
def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.75, 1.33)))
    else:
        transforms.append(T.Resize((224, 224)))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


def get_model_instance_segmentation(num_classes):
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def generate_confusion_matrix(data_loader_test, model, device, num_classes, output_folder):
    all_true_labels = []
    all_pred_labels = []

    model.to(device)
    model.eval()

    for images, targets in data_loader_test:
        # Envoi sur GPU
        images = [img.to(device) for img in images]
        processed_targets = []

        for t in targets:
            converted_target = {}
            for k, v in t.items():
                if isinstance(v, (int, float, np.integer, np.floating)):
                    converted_target[k] = torch.tensor(v, device=device)
                elif isinstance(v, np.ndarray):
                    converted_target[k] = torch.tensor(v, device=device)
                elif isinstance(v, torch.Tensor):
                    converted_target[k] = v.to(device)
                else:
                    raise ValueError(f"Type non pris en charge pour {k}: {type(v)}")
            processed_targets.append(converted_target)

        targets = processed_targets

        with torch.no_grad():
            outputs = model(images)

        # Prendre la première image du batch
        pred = outputs[0]
        true = targets[0]

        # print(f'Prédiction du modèle : {pred}')

        # Si aucune prédiction
        if len(pred['labels']) == 0:
            pred_label = 0  # On attribue une classe "0" pour aucun objet détecté
        else:
            idx_best = pred['scores'].argmax()
            pred_label = pred['labels'][idx_best].item()

        # Si aucune annotation réelle
        if len(true['labels']) == 0:
            true_label = 0  # On attribue une classe "0" pour aucune annotation
        else:
            true_label = true['labels'][0].item()

        all_true_labels.append(true_label)
        all_pred_labels.append(pred_label)

    # Convertir en numpy pour compatibilité avec scikit-learn
    all_true_labels = np.array(all_true_labels)
    all_pred_labels = np.array(all_pred_labels)

    # Générer la matrice de confusion
    conf_mat = confusion_matrix(all_true_labels, all_pred_labels, labels=range(num_classes))

    # Normalisation pour obtenir des pourcentages
    with np.errstate(divide='ignore', invalid='ignore'):
        conf_mat_pct = np.nan_to_num(
            conf_mat.astype(float) / conf_mat.sum(axis=1, keepdims=True),
            nan=0.0
        )

    # Sauvegarde de la matrice de confusion
    os.makedirs(output_folder, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat_pct, annot=True, cmap="Blues", fmt=".2f",
                xticklabels=range(num_classes),
                yticklabels=range(num_classes))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (%)")
    plt.savefig(os.path.join(output_folder, "confusion_matrix.png"))
    plt.close()

    print("Confusion matrix saved in", output_folder)



############################################################
# (E) FONCTION D'ENTRAÎNEMENT PRINCIPALE
############################################################

def train_model(
    scans_folder,
    masks_folder,
    temp_folder,
    slices_csv,
    num_epochs,
    lr,
    device,
    output_folder
):
    # 1) Extraction (désactivée ici si le CSV est déjà créé)
    # print("=== Étape 1 : Extraction & sauvegarde des slices ===")
    # extract_and_save_slices_optimized(
    #     scans_folder,
    #     masks_folder,
    #     temp_folder,
    #     slices_csv,
    #     min_box_size=20,  # À ajuster si nécessaire
    #     num_workers=4    # Ajustez selon vos ressources
    # )

    # 2) Création du Dataset
    print("=== Étape 2 : Création du Dataset ===")
    dataset_full = TempSlicesDataset(slices_csv, transforms=None)
    full_size = len(dataset_full)
    print(f"[INFO] dataset_full size: {full_size}")

    # 3) Split train / val
    indices = np.arange(full_size)
    np.random.shuffle(indices)
    train_split = int(0.95 * full_size)
    train_idx = indices[:train_split]
    val_idx   = indices[train_split:]

    train_dataset = torch.utils.data.Subset(dataset_full, train_idx)
    val_dataset   = torch.utils.data.Subset(dataset_full, val_idx)

    print("=== Étape 3 : DataLoaders ===")
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=utils.collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=utils.collate_fn,
        num_workers=0
    )

    # 4) Modèle
    print("=== Étape 4 : Instanciation du modèle ===")
    num_classes = 4  # Fond + 3 classes
    model = get_model_instance_segmentation(num_classes).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=0.0005
    )
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # 5) Entraînement
    print("=== Étape 5 : Entraînement ===")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=1)
        evaluate(model, val_loader, device=device)
        lr_scheduler.step()

    os.makedirs(output_folder, exist_ok=True)
    model_path = os.path.join(output_folder, "final_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Modèle final sauvegardé : {model_path}")

    # 6) Matrice de confusion et évaluation finale
    print("=== Étape 6 : Évaluation Test & Matrice de confusion ===")
    model.eval()
    generate_confusion_matrix(
        val_loader,
        model,
        device,
        num_classes=num_classes,
        output_folder=output_folder
    )
    evaluate(model, val_loader, device=device)

    return model_path


############################################################
# (F) MAIN
############################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Mask R-CNN model with valid NIfTI slices (extracted to .npy) from axial, coronal, and sagittal views."
    )
    parser.add_argument('--scans_folder', type=str, required=True,
                        help='Répertoire contenant les fichiers scan .nii.')
    parser.add_argument('--masks_folder', type=str, required=True,
                        help='Répertoire contenant les fichiers masque .nii.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Nombre total d\'époques.')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Dossier pour sauvegarder le modèle et résultats.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--temp_folder', type=str, default='temp_slices_resample_v4',
                        help='Dossier pour stocker les slices .npy.')
    parser.add_argument('--csv_path', type=str, default='slices_metadata_resample_v4.csv',
                        help='Fichier CSV listant les slices.')
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    train_model(
        scans_folder=args.scans_folder,
        masks_folder=args.masks_folder,
        temp_folder=args.temp_folder,
        slices_csv=args.csv_path,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        output_folder=args.output_folder
    )
