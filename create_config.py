import os
import json

# make dictionary with parameters specific to all datasets
data = {}
data['isic'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC',
  'output_path': 'output_ISIC'
  }
data['isic_2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_2',
  'output_path': 'output_ISIC_2'
  }
data['ISIC_image_rot_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_image_rot_f=0.1',
  'output_path': 'output_ISIC_ISIC_image_rot_f=0.1'
  }
data['ISIC_image_rot_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_image_rot_f=0.2',
  'output_path': 'output_ISIC_ISIC_image_rot_f=0.2'
  }
data['ISIC_image_rot_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_image_rot_f=0.3',
  'output_path': 'output_ISIC_ISIC_image_rot_f=0.3'
  }
data['ISIC_image_rot_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_image_rot_f=0.4',
  'output_path': 'output_ISIC_ISIC_image_rot_f=0.4'
  }
data['ISIC_image_rot_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_image_rot_f=0.5',
  'output_path': 'output_ISIC_ISIC_image_rot_f=0.5'
  }
data['ISIC_image_rot_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_image_rot_f=0.6',
  'output_path': 'output_ISIC_ISIC_image_rot_f=0.6'
  }
data['ISIC_image_rot_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_image_rot_f=0.7',
  'output_path': 'output_ISIC_ISIC_image_rot_f=0.7'
  }
data['ISIC_image_rot_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_image_rot_f=0.8',
  'output_path': 'output_ISIC_ISIC_image_rot_f=0.8'
  }
data['ISIC_image_rot_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_image_rot_f=0.9',
  'output_path': 'output_ISIC_ISIC_image_rot_f=0.9'
  }
data['ISIC_image_rot_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_image_rot_f=1.0',
  'output_path': 'output_ISIC_ISIC_image_rot_f=1.0'
  }

data['ISIC_image_translation_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_image_translation_f=0.1',
  'output_path': 'output_ISIC_ISIC_image_translation_f=0.1'
  }
data['ISIC_image_translation_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_image_translation_f=0.2',
  'output_path': 'output_ISIC_ISIC_image_translation_f=0.2'
  }
data['ISIC_image_translation_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_image_translation_f=0.3',
  'output_path': 'output_ISIC_ISIC_image_translation_f=0.3'
  }
data['ISIC_image_translation_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_image_translation_f=0.4',
  'output_path': 'output_ISIC_ISIC_image_translation_f=0.4'
  }
data['ISIC_image_translation_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_image_translation_f=0.5',
  'output_path': 'output_ISIC_ISIC_image_translation_f=0.5'
  }
data['ISIC_image_translation_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_image_translation_f=0.6',
  'output_path': 'output_ISIC_ISIC_image_translation_f=0.6'
  }
data['ISIC_image_translation_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_image_translation_f=0.7',
  'output_path': 'output_ISIC_ISIC_image_translation_f=0.7'
  }
data['ISIC_image_translation_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_image_translation_f=0.8',
  'output_path': 'output_ISIC_ISIC_image_translation_f=0.8'
  }
data['ISIC_image_translation_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_image_translation_f=0.9',
  'output_path': 'output_ISIC_ISIC_image_translation_f=0.9'
  }
data['ISIC_image_translation_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_image_translation_f=1.0',
  'output_path': 'output_ISIC_ISIC_image_translation_f=1.0'
  }

data['ISIC_image_zoom_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_image_zoom_f=0.1',
  'output_path': 'output_ISIC_ISIC_image_zoom_f=0.1'
  }
data['ISIC_image_zoom_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_image_zoom_f=0.2',
  'output_path': 'output_ISIC_ISIC_image_zoom_f=0.2'
  }
data['ISIC_image_zoom_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_image_zoom_f=0.3',
  'output_path': 'output_ISIC_ISIC_image_zoom_f=0.3'
  }
data['ISIC_image_zoom_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_image_zoom_f=0.4',
  'output_path': 'output_ISIC_ISIC_image_zoom_f=0.4'
  }
data['ISIC_image_zoom_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_image_zoom_f=0.5',
  'output_path': 'output_ISIC_ISIC_image_zoom_f=0.5'
  }
data['ISIC_image_zoom_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_image_zoom_f=0.6',
  'output_path': 'output_ISIC_ISIC_image_zoom_f=0.6'
  }
data['ISIC_image_zoom_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_image_zoom_f=0.7',
  'output_path': 'output_ISIC_ISIC_image_zoom_f=0.7'
  }
data['ISIC_image_zoom_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_image_zoom_f=0.8',
  'output_path': 'output_ISIC_ISIC_image_zoom_f=0.8'
  }
data['ISIC_image_zoom_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_image_zoom_f=0.9',
  'output_path': 'output_ISIC_ISIC_image_zoom_f=0.9'
  }
data['ISIC_image_zoom_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_image_zoom_f=1.0',
  'output_path': 'output_ISIC_ISIC_image_zoom_f=1.0'
  }

data['ISIC_add_noise_gaussian_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_gaussian_f=0.1',
  'output_path': 'output_ISIC_ISIC_add_noise_gaussian_f=0.1'
  }
data['ISIC_add_noise_gaussian_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_gaussian_f=0.2',
  'output_path': 'output_ISIC_ISIC_add_noise_gaussian_f=0.2'
  }
data['ISIC_add_noise_gaussian_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_gaussian_f=0.3',
  'output_path': 'output_ISIC_ISIC_add_noise_gaussian_f=0.3'
  }
data['ISIC_add_noise_gaussian_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_gaussian_f=0.4',
  'output_path': 'output_ISIC_ISIC_add_noise_gaussian_f=0.4'
  }
data['ISIC_add_noise_gaussian_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_gaussian_f=0.5',
  'output_path': 'output_ISIC_ISIC_add_noise_gaussian_f=0.5'
  }
data['ISIC_add_noise_gaussian_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_gaussian_f=0.6',
  'output_path': 'output_ISIC_ISIC_add_noise_gaussian_f=0.6'
  }
data['ISIC_add_noise_gaussian_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_gaussian_f=0.7',
  'output_path': 'output_ISIC_ISIC_add_noise_gaussian_f=0.7'
  }
data['ISIC_add_noise_gaussian_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_gaussian_f=0.8',
  'output_path': 'output_ISIC_ISIC_add_noise_gaussian_f=0.8'
  }
data['ISIC_add_noise_gaussian_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_gaussian_f=0.9',
  'output_path': 'output_ISIC_ISIC_add_noise_gaussian_f=0.9'
  }
data['ISIC_add_noise_gaussian_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_gaussian_f=1.0',
  'output_path': 'output_ISIC_ISIC_add_noise_gaussian_f=1.0'
  }

data['ISIC_add_noise_poisson_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_poisson_f=0.1',
  'output_path': 'output_ISIC_ISIC_add_noise_poisson_f=0.1'
  }
data['ISIC_add_noise_poisson_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_poisson_f=0.2',
  'output_path': 'output_ISIC_ISIC_add_noise_poisson_f=0.2'
  }
data['ISIC_add_noise_poisson_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_poisson_f=0.3',
  'output_path': 'output_ISIC_ISIC_add_noise_poisson_f=0.3'
  }
data['ISIC_add_noise_poisson_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_poisson_f=0.4',
  'output_path': 'output_ISIC_ISIC_add_noise_poisson_f=0.4'
  }
data['ISIC_add_noise_poisson_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_poisson_f=0.5',
  'output_path': 'output_ISIC_ISIC_add_noise_poisson_f=0.5'
  }
data['ISIC_add_noise_poisson_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_poisson_f=0.6',
  'output_path': 'output_ISIC_ISIC_add_noise_poisson_f=0.6'
  }
data['ISIC_add_noise_poisson_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_poisson_f=0.7',
  'output_path': 'output_ISIC_ISIC_add_noise_poisson_f=0.7'
  }
data['ISIC_add_noise_poisson_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_poisson_f=0.8',
  'output_path': 'output_ISIC_ISIC_add_noise_poisson_f=0.8'
  }
data['ISIC_add_noise_poisson_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_poisson_f=0.9',
  'output_path': 'output_ISIC_ISIC_add_noise_poisson_f=0.9'
  }
data['ISIC_add_noise_poisson_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_poisson_f=1.0',
  'output_path': 'output_ISIC_ISIC_add_noise_poisson_f=1.0'
  }

data['ISIC_add_noise_salt_and_pepper_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_salt_and_pepper_f=0.1',
  'output_path': 'output_ISIC_ISIC_add_noise_salt_and_pepper_f=0.1'
  }
data['ISIC_add_noise_salt_and_pepper_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_salt_and_pepper_f=0.2',
  'output_path': 'output_ISIC_ISIC_add_noise_salt_and_pepper_f=0.2'
  }
data['ISIC_add_noise_salt_and_pepper_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_salt_and_pepper_f=0.3',
  'output_path': 'output_ISIC_ISIC_add_noise_salt_and_pepper_f=0.3'
  }
data['ISIC_add_noise_salt_and_pepper_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_salt_and_pepper_f=0.4',
  'output_path': 'output_ISIC_ISIC_add_noise_salt_and_pepper_f=0.4'
  }
data['ISIC_add_noise_salt_and_pepper_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_salt_and_pepper_f=0.5',
  'output_path': 'output_ISIC_ISIC_add_noise_salt_and_pepper_f=0.5'
  }
data['ISIC_add_noise_salt_and_pepper_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_salt_and_pepper_f=0.6',
  'output_path': 'output_ISIC_ISIC_add_noise_salt_and_pepper_f=0.6'
  }
data['ISIC_add_noise_salt_and_pepper_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_salt_and_pepper_f=0.7',
  'output_path': 'output_ISIC_ISIC_add_noise_salt_and_pepper_f=0.7'
  }
data['ISIC_add_noise_salt_and_pepper_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_salt_and_pepper_f=0.8',
  'output_path': 'output_ISIC_ISIC_add_noise_salt_and_pepper_f=0.8'
  }
data['ISIC_add_noise_salt_and_pepper_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_salt_and_pepper_f=0.9',
  'output_path': 'output_ISIC_ISIC_add_noise_salt_and_pepper_f=0.9'
  }
data['ISIC_add_noise_salt_and_pepper_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_salt_and_pepper_f=1.0',
  'output_path': 'output_ISIC_ISIC_add_noise_salt_and_pepper_f=1.0'
  }

data['ISIC_add_noise_speckle_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_speckle_f=0.1',
  'output_path': 'output_ISIC_ISIC_add_noise_speckle_f=0.1'
  }
data['ISIC_add_noise_speckle_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_speckle_f=0.2',
  'output_path': 'output_ISIC_ISIC_add_noise_speckle_f=0.2'
  }
data['ISIC_add_noise_speckle_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_speckle_f=0.3',
  'output_path': 'output_ISIC_ISIC_add_noise_speckle_f=0.3'
  }
data['ISIC_add_noise_speckle_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_speckle_f=0.4',
  'output_path': 'output_ISIC_ISIC_add_noise_speckle_f=0.4'
  }
data['ISIC_add_noise_speckle_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_speckle_f=0.5',
  'output_path': 'output_ISIC_ISIC_add_noise_speckle_f=0.5'
  }
data['ISIC_add_noise_speckle_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_speckle_f=0.6',
  'output_path': 'output_ISIC_ISIC_add_noise_speckle_f=0.6'
  }
data['ISIC_add_noise_speckle_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_speckle_f=0.7',
  'output_path': 'output_ISIC_ISIC_add_noise_speckle_f=0.7'
  }
data['ISIC_add_noise_speckle_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_speckle_f=0.8',
  'output_path': 'output_ISIC_ISIC_add_noise_speckle_f=0.8'
  }
data['ISIC_add_noise_speckle_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_speckle_f=0.9',
  'output_path': 'output_ISIC_ISIC_add_noise_speckle_f=0.9'
  }
data['ISIC_add_noise_speckle_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC',
  'dataset_path': 'dataset_ISIC_add_noise_speckle_f=1.0',
  'output_path': 'output_ISIC_ISIC_add_noise_speckle_f=1.0'
  }

data['ISIC_imbalance_classes_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC',
'dataset_path': 'dataset_ISIC_imbalance_classes_f=0.1',
'output_path': 'output_ISIC_ISIC_imbalance_classes_f=0.1'
}
data['ISIC_imbalance_classes_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC',
'dataset_path': 'dataset_ISIC_imbalance_classes_f=0.2',
'output_path': 'output_ISIC_ISIC_imbalance_classes_f=0.2'
}
data['ISIC_imbalance_classes_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC',
'dataset_path': 'dataset_ISIC_imbalance_classes_f=0.3',
'output_path': 'output_ISIC_ISIC_imbalance_classes_f=0.3'
}
data['ISIC_imbalance_classes_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC',
'dataset_path': 'dataset_ISIC_imbalance_classes_f=0.4',
'output_path': 'output_ISIC_ISIC_imbalance_classes_f=0.4'
}
data['ISIC_imbalance_classes_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC',
'dataset_path': 'dataset_ISIC_imbalance_classes_f=0.5',
'output_path': 'output_ISIC_ISIC_imbalance_classes_f=0.5'
}
data['ISIC_imbalance_classes_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC',
'dataset_path': 'dataset_ISIC_imbalance_classes_f=0.6',
'output_path': 'output_ISIC_ISIC_imbalance_classes_f=0.6'
}
data['ISIC_imbalance_classes_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC',
'dataset_path': 'dataset_ISIC_imbalance_classes_f=0.7',
'output_path': 'output_ISIC_ISIC_imbalance_classes_f=0.7'
}
data['ISIC_imbalance_classes_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC',
'dataset_path': 'dataset_ISIC_imbalance_classes_f=0.8',
'output_path': 'output_ISIC_ISIC_imbalance_classes_f=0.8'
}
data['ISIC_imbalance_classes_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC',
'dataset_path': 'dataset_ISIC_imbalance_classes_f=0.9',
'output_path': 'output_ISIC_ISIC_imbalance_classes_f=0.9'
}
data['ISIC_imbalance_classes_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC',
'dataset_path': 'dataset_ISIC_imbalance_classes_f=1.0',
'output_path': 'output_ISIC_ISIC_imbalance_classes_f=1.0'
}

data['isic_2017'] = {
  'classes': ['melanoma', 'nevus_sk'],
  'orig_path': 'ISIC-2017',
  'dataset_path': 'dataset_ISIC_2017',
  'output_path': 'output_ISIC_2017'
  }
data['isic_2017_adj'] = {
  'classes': ['melanoma', 'nevus_sk'],
  'orig_path': 'ISIC_2017_adj',
  'dataset_path': 'dataset_ISIC_adj',
  'output_path': 'output_ISIC_adj'
  }
data['cats_and_dogs'] = {
  'classes': ['cat', 'dog'],
  'orig_path': 'cats_and_dogs',
  'dataset_path': 'dataset_cats_and_dogs',
  'output_path': 'output_cats_and_dogs'
  }

# list with all datasets; so certain paths can be added automatically to the data dictionary
datasets = ['isic',
            'isic_2',
            'ISIC_image_rot_f=0.1',
            'ISIC_image_rot_f=0.2',
            'ISIC_image_rot_f=0.3',
            'ISIC_image_rot_f=0.4',
            'ISIC_image_rot_f=0.5',
            'ISIC_image_rot_f=0.6',
            'ISIC_image_rot_f=0.7',
            'ISIC_image_rot_f=0.8',
            'ISIC_image_rot_f=0.9',
            'ISIC_image_rot_f=1.0',
            'ISIC_image_translation_f=0.1',
            'ISIC_image_translation_f=0.2',
            'ISIC_image_translation_f=0.3',
            'ISIC_image_translation_f=0.4',
            'ISIC_image_translation_f=0.5',
            'ISIC_image_translation_f=0.6',
            'ISIC_image_translation_f=0.7',
            'ISIC_image_translation_f=0.8',
            'ISIC_image_translation_f=0.9',
            'ISIC_image_translation_f=1.0',
            'ISIC_image_zoom_f=0.1',
            'ISIC_image_zoom_f=0.2',
            'ISIC_image_zoom_f=0.3',
            'ISIC_image_zoom_f=0.4',
            'ISIC_image_zoom_f=0.5',
            'ISIC_image_zoom_f=0.6',
            'ISIC_image_zoom_f=0.7',
            'ISIC_image_zoom_f=0.8',
            'ISIC_image_zoom_f=0.9',
            'ISIC_image_zoom_f=1.0',
            'ISIC_add_noise_gaussian_f=0.1',
            'ISIC_add_noise_gaussian_f=0.2',
            'ISIC_add_noise_gaussian_f=0.3',
            'ISIC_add_noise_gaussian_f=0.4',
            'ISIC_add_noise_gaussian_f=0.5',
            'ISIC_add_noise_gaussian_f=0.6',
            'ISIC_add_noise_gaussian_f=0.7',
            'ISIC_add_noise_gaussian_f=0.8',
            'ISIC_add_noise_gaussian_f=0.9',
            'ISIC_add_noise_gaussian_f=1.0',
            'ISIC_add_noise_poisson_f=0.1',
            'ISIC_add_noise_poisson_f=0.2',
            'ISIC_add_noise_poisson_f=0.3',
            'ISIC_add_noise_poisson_f=0.4',
            'ISIC_add_noise_poisson_f=0.5',
            'ISIC_add_noise_poisson_f=0.6',
            'ISIC_add_noise_poisson_f=0.7',
            'ISIC_add_noise_poisson_f=0.8',
            'ISIC_add_noise_poisson_f=0.9',
            'ISIC_add_noise_poisson_f=1.0',
            'ISIC_add_noise_salt_and_pepper_f=0.1',
            'ISIC_add_noise_salt_and_pepper_f=0.2',
            'ISIC_add_noise_salt_and_pepper_f=0.3',
            'ISIC_add_noise_salt_and_pepper_f=0.4',
            'ISIC_add_noise_salt_and_pepper_f=0.5',
            'ISIC_add_noise_salt_and_pepper_f=0.6',
            'ISIC_add_noise_salt_and_pepper_f=0.7',
            'ISIC_add_noise_salt_and_pepper_f=0.8',
            'ISIC_add_noise_salt_and_pepper_f=0.9',
            'ISIC_add_noise_salt_and_pepper_f=1.0',
            'ISIC_add_noise_speckle_f=0.1',
            'ISIC_add_noise_speckle_f=0.2',
            'ISIC_add_noise_speckle_f=0.3',
            'ISIC_add_noise_speckle_f=0.4',
            'ISIC_add_noise_speckle_f=0.5',
            'ISIC_add_noise_speckle_f=0.6',
            'ISIC_add_noise_speckle_f=0.7',
            'ISIC_add_noise_speckle_f=0.8',
            'ISIC_add_noise_speckle_f=0.9',
            'ISIC_add_noise_speckle_f=1.0',
            'ISIC_imbalance_classes_f=0.1',
            'ISIC_imbalance_classes_f=0.2',
            'ISIC_imbalance_classes_f=0.3',
            'ISIC_imbalance_classes_f=0.4',
            'ISIC_imbalance_classes_f=0.5',
            'ISIC_imbalance_classes_f=0.6',
            'ISIC_imbalance_classes_f=0.7',
            'ISIC_imbalance_classes_f=0.8',
            'ISIC_imbalance_classes_f=0.9',
            'ISIC_imbalance_classes_f=1.0',
            'isic_2017',
            'isic_2017_adj',
            'cats_and_dogs']

for dataset in datasets:
    # path to original dataset, use parent path so data is not in same repo as code
    parent_path = os.path.dirname(os.getcwd())
    orig_data_path = os.path.join(parent_path, 'Data/{}'.format(data[dataset]['orig_path']))
    data[dataset]['orig_data_path'] = orig_data_path

    # base path after splitting data
    dataset_path = os.path.join(parent_path, 'datasets/{}'.format(data[dataset]['dataset_path']))
    data[dataset]['dataset_path'] = dataset_path

    # path to save plots
    plot_path = os.path.join(parent_path, 'outputs/{}/plots'.format(data[dataset]['output_path']))
    data[dataset]['plot_path'] = plot_path

    # path to save trained model
    model_savepath = os.path.join(parent_path, 'outputs/{}/models'.format(data[dataset]['output_path']))
    data[dataset]['model_savepath'] = model_savepath

    # data split paths
    trainingpath = os.path.join(dataset_path, 'training')
    data[dataset]['trainingpath'] = trainingpath
    validationpath = os.path.join(dataset_path, 'validation')
    data[dataset]['validationpath'] = validationpath
    testpath = os.path.join(dataset_path, 'test')
    data[dataset]['testpath'] = testpath

# create json configuration file
with open('config.json', 'w') as f:
    json.dump(data, f)
