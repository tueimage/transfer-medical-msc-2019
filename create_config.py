import os
import json

# make dictionary with parameters specific to all datasets
data = {}
data['ddsm'] = {
    'classes': ['positive', 'negative'],
    'orig_path': 'ddsm-mammography',
    'dataset_path': 'dataset_ddsm',
    'output_path': 'output_ddsm'
}
data['CNMC'] = {
    'classes': ['normal', 'leukemic'],
    'orig_path': 'C-NMC_Leukemia',
    'dataset_path': 'dataset_CNMC',
    'output_path': 'output_CNMC'
}
data['ISIC_2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2',
  'output_path': 'output_ISIC_2'
  }
data['ISIC_2_image_rot_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_image_rot_f=0.1',
  'output_path': 'output_ISIC_2_image_rot_f=0.1'
  }
data['ISIC_2_image_rot_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_image_rot_f=0.2',
  'output_path': 'output_ISIC_2_image_rot_f=0.2'
  }
data['ISIC_2_image_rot_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_image_rot_f=0.3',
  'output_path': 'output_ISIC_2_image_rot_f=0.3'
  }
data['ISIC_2_image_rot_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_image_rot_f=0.4',
  'output_path': 'output_ISIC_2_image_rot_f=0.4'
  }
data['ISIC_2_image_rot_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_image_rot_f=0.5',
  'output_path': 'output_ISIC_2_image_rot_f=0.5'
  }
data['ISIC_2_image_rot_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_image_rot_f=0.6',
  'output_path': 'output_ISIC_2_image_rot_f=0.6'
  }
data['ISIC_2_image_rot_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_image_rot_f=0.7',
  'output_path': 'output_ISIC_2_image_rot_f=0.7'
  }
data['ISIC_2_image_rot_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_image_rot_f=0.8',
  'output_path': 'output_ISIC_2_image_rot_f=0.8'
  }
data['ISIC_2_image_rot_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_image_rot_f=0.9',
  'output_path': 'output_ISIC_2_image_rot_f=0.9'
  }
data['ISIC_2_image_rot_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_image_rot_f=1.0',
  'output_path': 'output_ISIC_2_image_rot_f=1.0'
  }

data['ISIC_2_image_translation_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_image_translation_f=0.1',
  'output_path': 'output_ISIC_2_image_translation_f=0.1'
  }
data['ISIC_2_image_translation_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_image_translation_f=0.2',
  'output_path': 'output_ISIC_2_image_translation_f=0.2'
  }
data['ISIC_2_image_translation_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_image_translation_f=0.3',
  'output_path': 'output_ISIC_2_image_translation_f=0.3'
  }
data['ISIC_2_image_translation_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_image_translation_f=0.4',
  'output_path': 'output_ISIC_2_image_translation_f=0.4'
  }
data['ISIC_2_image_translation_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_image_translation_f=0.5',
  'output_path': 'output_ISIC_2_image_translation_f=0.5'
  }
data['ISIC_2_image_translation_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_image_translation_f=0.6',
  'output_path': 'output_ISIC_2_image_translation_f=0.6'
  }
data['ISIC_2_image_translation_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_image_translation_f=0.7',
  'output_path': 'output_ISIC_2_image_translation_f=0.7'
  }
data['ISIC_2_image_translation_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_image_translation_f=0.8',
  'output_path': 'output_ISIC_2_image_translation_f=0.8'
  }
data['ISIC_2_image_translation_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_image_translation_f=0.9',
  'output_path': 'output_ISIC_2_image_translation_f=0.9'
  }
data['ISIC_2_image_translation_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_image_translation_f=1.0',
  'output_path': 'output_ISIC_2_image_translation_f=1.0'
  }

data['ISIC_2_image_zoom_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_image_zoom_f=0.1',
  'output_path': 'output_ISIC_2_image_zoom_f=0.1'
  }
data['ISIC_2_image_zoom_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_image_zoom_f=0.2',
  'output_path': 'output_ISIC_2_image_zoom_f=0.2'
  }
data['ISIC_2_image_zoom_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_image_zoom_f=0.3',
  'output_path': 'output_ISIC_2_image_zoom_f=0.3'
  }
data['ISIC_2_image_zoom_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_image_zoom_f=0.4',
  'output_path': 'output_ISIC_2_image_zoom_f=0.4'
  }
data['ISIC_2_image_zoom_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_image_zoom_f=0.5',
  'output_path': 'output_ISIC_2_image_zoom_f=0.5'
  }
data['ISIC_2_image_zoom_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_image_zoom_f=0.6',
  'output_path': 'output_ISIC_2_image_zoom_f=0.6'
  }
data['ISIC_2_image_zoom_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_image_zoom_f=0.7',
  'output_path': 'output_ISIC_2_image_zoom_f=0.7'
  }
data['ISIC_2_image_zoom_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_image_zoom_f=0.8',
  'output_path': 'output_ISIC_2_image_zoom_f=0.8'
  }
data['ISIC_2_image_zoom_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_image_zoom_f=0.9',
  'output_path': 'output_ISIC_2_image_zoom_f=0.9'
  }
data['ISIC_2_image_zoom_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_image_zoom_f=1.0',
  'output_path': 'output_ISIC_2_image_zoom_f=1.0'
  }

data['ISIC_2_add_noise_gaussian_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_gaussian_f=0.1',
  'output_path': 'output_ISIC_2_add_noise_gaussian_f=0.1'
  }
data['ISIC_2_add_noise_gaussian_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_gaussian_f=0.2',
  'output_path': 'output_ISIC_2_add_noise_gaussian_f=0.2'
  }
data['ISIC_2_add_noise_gaussian_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_gaussian_f=0.3',
  'output_path': 'output_ISIC_2_add_noise_gaussian_f=0.3'
  }
data['ISIC_2_add_noise_gaussian_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_gaussian_f=0.4',
  'output_path': 'output_ISIC_2_add_noise_gaussian_f=0.4'
  }
data['ISIC_2_add_noise_gaussian_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_gaussian_f=0.5',
  'output_path': 'output_ISIC_2_add_noise_gaussian_f=0.5'
  }
data['ISIC_2_add_noise_gaussian_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_gaussian_f=0.6',
  'output_path': 'output_ISIC_2_add_noise_gaussian_f=0.6'
  }
data['ISIC_2_add_noise_gaussian_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_gaussian_f=0.7',
  'output_path': 'output_ISIC_2_add_noise_gaussian_f=0.7'
  }
data['ISIC_2_add_noise_gaussian_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_gaussian_f=0.8',
  'output_path': 'output_ISIC_2_add_noise_gaussian_f=0.8'
  }
data['ISIC_2_add_noise_gaussian_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_gaussian_f=0.9',
  'output_path': 'output_ISIC_2_add_noise_gaussian_f=0.9'
  }
data['ISIC_2_add_noise_gaussian_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_gaussian_f=1.0',
  'output_path': 'output_ISIC_2_add_noise_gaussian_f=1.0'
  }

data['ISIC_2_add_noise_poisson_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_poisson_f=0.1',
  'output_path': 'output_ISIC_2_add_noise_poisson_f=0.1'
  }
data['ISIC_2_add_noise_poisson_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_poisson_f=0.2',
  'output_path': 'output_ISIC_2_add_noise_poisson_f=0.2'
  }
data['ISIC_2_add_noise_poisson_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_poisson_f=0.3',
  'output_path': 'output_ISIC_2_add_noise_poisson_f=0.3'
  }
data['ISIC_2_add_noise_poisson_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_poisson_f=0.4',
  'output_path': 'output_ISIC_2_add_noise_poisson_f=0.4'
  }
data['ISIC_2_add_noise_poisson_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_poisson_f=0.5',
  'output_path': 'output_ISIC_2_add_noise_poisson_f=0.5'
  }
data['ISIC_2_add_noise_poisson_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_poisson_f=0.6',
  'output_path': 'output_ISIC_2_add_noise_poisson_f=0.6'
  }
data['ISIC_2_add_noise_poisson_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_poisson_f=0.7',
  'output_path': 'output_ISIC_2_add_noise_poisson_f=0.7'
  }
data['ISIC_2_add_noise_poisson_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_poisson_f=0.8',
  'output_path': 'output_ISIC_2_add_noise_poisson_f=0.8'
  }
data['ISIC_2_add_noise_poisson_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_poisson_f=0.9',
  'output_path': 'output_ISIC_2_add_noise_poisson_f=0.9'
  }
data['ISIC_2_add_noise_poisson_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_poisson_f=1.0',
  'output_path': 'output_ISIC_2_add_noise_poisson_f=1.0'
  }

data['ISIC_2_add_noise_salt_and_pepper_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_salt_and_pepper_f=0.1',
  'output_path': 'output_ISIC_2_add_noise_salt_and_pepper_f=0.1'
  }
data['ISIC_2_add_noise_salt_and_pepper_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_salt_and_pepper_f=0.2',
  'output_path': 'output_ISIC_2_add_noise_salt_and_pepper_f=0.2'
  }
data['ISIC_2_add_noise_salt_and_pepper_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_salt_and_pepper_f=0.3',
  'output_path': 'output_ISIC_2_add_noise_salt_and_pepper_f=0.3'
  }
data['ISIC_2_add_noise_salt_and_pepper_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_salt_and_pepper_f=0.4',
  'output_path': 'output_ISIC_2_add_noise_salt_and_pepper_f=0.4'
  }
data['ISIC_2_add_noise_salt_and_pepper_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_salt_and_pepper_f=0.5',
  'output_path': 'output_ISIC_2_add_noise_salt_and_pepper_f=0.5'
  }
data['ISIC_2_add_noise_salt_and_pepper_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_salt_and_pepper_f=0.6',
  'output_path': 'output_ISIC_2_add_noise_salt_and_pepper_f=0.6'
  }
data['ISIC_2_add_noise_salt_and_pepper_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_salt_and_pepper_f=0.7',
  'output_path': 'output_ISIC_2_add_noise_salt_and_pepper_f=0.7'
  }
data['ISIC_2_add_noise_salt_and_pepper_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_salt_and_pepper_f=0.8',
  'output_path': 'output_ISIC_2_add_noise_salt_and_pepper_f=0.8'
  }
data['ISIC_2_add_noise_salt_and_pepper_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_salt_and_pepper_f=0.9',
  'output_path': 'output_ISIC_2_add_noise_salt_and_pepper_f=0.9'
  }
data['ISIC_2_add_noise_salt_and_pepper_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_salt_and_pepper_f=1.0',
  'output_path': 'output_ISIC_2_add_noise_salt_and_pepper_f=1.0'
  }

data['ISIC_2_add_noise_speckle_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_speckle_f=0.1',
  'output_path': 'output_ISIC_2_add_noise_speckle_f=0.1'
  }
data['ISIC_2_add_noise_speckle_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_speckle_f=0.2',
  'output_path': 'output_ISIC_2_add_noise_speckle_f=0.2'
  }
data['ISIC_2_add_noise_speckle_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_speckle_f=0.3',
  'output_path': 'output_ISIC_2_add_noise_speckle_f=0.3'
  }
data['ISIC_2_add_noise_speckle_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_speckle_f=0.4',
  'output_path': 'output_ISIC_2_add_noise_speckle_f=0.4'
  }
data['ISIC_2_add_noise_speckle_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_speckle_f=0.5',
  'output_path': 'output_ISIC_2_add_noise_speckle_f=0.5'
  }
data['ISIC_2_add_noise_speckle_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_speckle_f=0.6',
  'output_path': 'output_ISIC_2_add_noise_speckle_f=0.6'
  }
data['ISIC_2_add_noise_speckle_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_speckle_f=0.7',
  'output_path': 'output_ISIC_2_add_noise_speckle_f=0.7'
  }
data['ISIC_2_add_noise_speckle_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_speckle_f=0.8',
  'output_path': 'output_ISIC_2_add_noise_speckle_f=0.8'
  }
data['ISIC_2_add_noise_speckle_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_speckle_f=0.9',
  'output_path': 'output_ISIC_2_add_noise_speckle_f=0.9'
  }
data['ISIC_2_add_noise_speckle_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_add_noise_speckle_f=1.0',
  'output_path': 'output_ISIC_2_add_noise_speckle_f=1.0'
  }

data['ISIC_2_imbalance_classes_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_imbalance_classes_f=0.1',
'output_path': 'output_ISIC_2_imbalance_classes_f=0.1'
}
data['ISIC_2_imbalance_classes_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_imbalance_classes_f=0.2',
'output_path': 'output_ISIC_2_imbalance_classes_f=0.2'
}
data['ISIC_2_imbalance_classes_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_imbalance_classes_f=0.3',
'output_path': 'output_ISIC_2_imbalance_classes_f=0.3'
}
data['ISIC_2_imbalance_classes_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_imbalance_classes_f=0.4',
'output_path': 'output_ISIC_2_imbalance_classes_f=0.4'
}
data['ISIC_2_imbalance_classes_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_imbalance_classes_f=0.5',
'output_path': 'output_ISIC_2_imbalance_classes_f=0.5'
}
data['ISIC_2_imbalance_classes_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_imbalance_classes_f=0.6',
'output_path': 'output_ISIC_2_imbalance_classes_f=0.6'
}
data['ISIC_2_imbalance_classes_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_imbalance_classes_f=0.7',
'output_path': 'output_ISIC_2_imbalance_classes_f=0.7'
}
data['ISIC_2_imbalance_classes_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_imbalance_classes_f=0.8',
'output_path': 'output_ISIC_2_imbalance_classes_f=0.8'
}
data['ISIC_2_imbalance_classes_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_imbalance_classes_f=0.9',
'output_path': 'output_ISIC_2_imbalance_classes_f=0.9'
}
data['ISIC_2_imbalance_classes_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_imbalance_classes_f=1.0',
'output_path': 'output_ISIC_2_imbalance_classes_f=1.0'
}

data['ISIC_2_grayscale_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_grayscale_f=0.1',
'output_path': 'output_ISIC_2_grayscale_f=0.1'
}
data['ISIC_2_grayscale_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_grayscale_f=0.2',
'output_path': 'output_ISIC_2_grayscale_f=0.2'
}
data['ISIC_2_grayscale_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_grayscale_f=0.3',
'output_path': 'output_ISIC_2_grayscale_f=0.3'
}
data['ISIC_2_grayscale_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_grayscale_f=0.4',
'output_path': 'output_ISIC_2_grayscale_f=0.4'
}
data['ISIC_2_grayscale_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_grayscale_f=0.5',
'output_path': 'output_ISIC_2_grayscale_f=0.5'
}
data['ISIC_2_grayscale_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_grayscale_f=0.6',
'output_path': 'output_ISIC_2_grayscale_f=0.6'
}
data['ISIC_2_grayscale_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_grayscale_f=0.7',
'output_path': 'output_ISIC_2_grayscale_f=0.7'
}
data['ISIC_2_grayscale_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_grayscale_f=0.8',
'output_path': 'output_ISIC_2_grayscale_f=0.8'
}
data['ISIC_2_grayscale_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_grayscale_f=0.9',
'output_path': 'output_ISIC_2_grayscale_f=0.9'
}
data['ISIC_2_grayscale_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_grayscale_f=1.0',
'output_path': 'output_ISIC_2_grayscale_f=1.0'
}

data['ISIC_2_hsv_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_hsv_f=0.1',
'output_path': 'output_ISIC_2_hsv_f=0.1'
}
data['ISIC_2_hsv_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_hsv_f=0.2',
'output_path': 'output_ISIC_2_hsv_f=0.2'
}
data['ISIC_2_hsv_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_hsv_f=0.3',
'output_path': 'output_ISIC_2_hsv_f=0.3'
}
data['ISIC_2_hsv_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_hsv_f=0.4',
'output_path': 'output_ISIC_2_hsv_f=0.4'
}
data['ISIC_2_hsv_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_hsv_f=0.5',
'output_path': 'output_ISIC_2_hsv_f=0.5'
}
data['ISIC_2_hsv_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_hsv_f=0.6',
'output_path': 'output_ISIC_2_hsv_f=0.6'
}
data['ISIC_2_hsv_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_hsv_f=0.7',
'output_path': 'output_ISIC_2_hsv_f=0.7'
}
data['ISIC_2_hsv_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_hsv_f=0.8',
'output_path': 'output_ISIC_2_hsv_f=0.8'
}
data['ISIC_2_hsv_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_hsv_f=0.9',
'output_path': 'output_ISIC_2_hsv_f=0.9'
}
data['ISIC_2_hsv_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_hsv_f=1.0',
'output_path': 'output_ISIC_2_hsv_f=1.0'
}
data['ISIC_2_blur_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_blur_f=0.1',
'output_path': 'output_ISIC_2_blur_f=0.1'
}
data['ISIC_2_blur_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_blur_f=0.2',
'output_path': 'output_ISIC_2_blur_f=0.2'
}
data['ISIC_2_blur_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_blur_f=0.3',
'output_path': 'output_ISIC_2_blur_f=0.3'
}
data['ISIC_2_blur_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_blur_f=0.4',
'output_path': 'output_ISIC_2_blur_f=0.4'
}
data['ISIC_2_blur_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_blur_f=0.5',
'output_path': 'output_ISIC_2_blur_f=0.5'
}
data['ISIC_2_blur_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_blur_f=0.6',
'output_path': 'output_ISIC_2_blur_f=0.6'
}
data['ISIC_2_blur_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_blur_f=0.7',
'output_path': 'output_ISIC_2_blur_f=0.7'
}
data['ISIC_2_blur_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_blur_f=0.8',
'output_path': 'output_ISIC_2_blur_f=0.8'
}
data['ISIC_2_blur_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_blur_f=0.9',
'output_path': 'output_ISIC_2_blur_f=0.9'
}
data['ISIC_2_blur_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_blur_f=1.0',
'output_path': 'output_ISIC_2_blur_f=1.0'
}

data['ISIC_2_small_random_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_small_random_f=0.1',
'output_path': 'output_ISIC_2_small_random_f=0.1'
}
data['ISIC_2_small_random_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_small_random_f=0.2',
'output_path': 'output_ISIC_2_small_random_f=0.2'
}
data['ISIC_2_small_random_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_small_random_f=0.3',
'output_path': 'output_ISIC_2_small_random_f=0.3'
}
data['ISIC_2_small_random_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_small_random_f=0.4',
'output_path': 'output_ISIC_2_small_random_f=0.4'
}
data['ISIC_2_small_random_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_small_random_f=0.5',
'output_path': 'output_ISIC_2_small_random_f=0.5'
}
data['ISIC_2_small_random_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_small_random_f=0.6',
'output_path': 'output_ISIC_2_small_random_f=0.6'
}
data['ISIC_2_small_random_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_small_random_f=0.7',
'output_path': 'output_ISIC_2_small_random_f=0.7'
}
data['ISIC_2_small_random_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_small_random_f=0.8',
'output_path': 'output_ISIC_2_small_random_f=0.8'
}
data['ISIC_2_small_random_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_small_random_f=0.9',
'output_path': 'output_ISIC_2_small_random_f=0.9'
}
data['ISIC_2_small_random_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_small_random_f=1.0',
'output_path': 'output_ISIC_2_small_random_f=1.0'
}

data['ISIC_2_small_easy_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_small_easy_f=0.1',
  'output_path': 'output_ISIC_2_small_easy_f=0.1'
  }
data['ISIC_2_small_easy_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_small_easy_f=0.2',
  'output_path': 'output_ISIC_2_small_easy_f=0.2'
  }
data['ISIC_2_small_easy_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_small_easy_f=0.3',
  'output_path': 'output_ISIC_2_small_easy_f=0.3'
  }
data['ISIC_2_small_easy_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_small_easy_f=0.4',
  'output_path': 'output_ISIC_2_small_easy_f=0.4'
  }
data['ISIC_2_small_easy_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_small_easy_f=0.5',
  'output_path': 'output_ISIC_2_small_easy_f=0.5'
  }
data['ISIC_2_small_easy_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_small_easy_f=0.6',
  'output_path': 'output_ISIC_2_small_easy_f=0.6'
  }
data['ISIC_2_small_easy_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_small_easy_f=0.7',
  'output_path': 'output_ISIC_2_small_easy_f=0.7'
  }
data['ISIC_2_small_easy_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_small_easy_f=0.8',
  'output_path': 'output_ISIC_2_small_easy_f=0.8'
  }
data['ISIC_2_small_easy_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_small_easy_f=0.9',
  'output_path': 'output_ISIC_2_small_easy_f=0.9'
  }
data['ISIC_2_small_easy_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_small_easy_f=1.0',
  'output_path': 'output_ISIC_2_small_easy_f=1.0'
  }

data['ISIC_2_small_hard_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_small_hard_f=0.1',
  'output_path': 'output_ISIC_2_small_hard_f=0.1'
  }
data['ISIC_2_small_hard_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_small_hard_f=0.2',
  'output_path': 'output_ISIC_2_small_hard_f=0.2'
  }
data['ISIC_2_small_hard_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_small_hard_f=0.3',
  'output_path': 'output_ISIC_2_small_hard_f=0.3'
  }
data['ISIC_2_small_hard_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_small_hard_f=0.4',
  'output_path': 'output_ISIC_2_small_hard_f=0.4'
  }
data['ISIC_2_small_hard_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_small_hard_f=0.5',
  'output_path': 'output_ISIC_2_small_hard_f=0.5'
  }
data['ISIC_2_small_hard_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_small_hard_f=0.6',
  'output_path': 'output_ISIC_2_small_hard_f=0.6'
  }
data['ISIC_2_small_hard_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_small_hard_f=0.7',
  'output_path': 'output_ISIC_2_small_hard_f=0.7'
  }
data['ISIC_2_small_hard_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_small_hard_f=0.8',
  'output_path': 'output_ISIC_2_small_hard_f=0.8'
  }
data['ISIC_2_small_hard_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_small_hard_f=0.9',
  'output_path': 'output_ISIC_2_small_hard_f=0.9'
  }
data['ISIC_2_small_hard_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_2',
  'dataset_path': 'dataset_ISIC_2_small_hard_f=1.0',
  'output_path': 'output_ISIC_2_small_hard_f=1.0'
  }

data['ISIC_2_small_clusters_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_small_clusters_f=0.1',
'output_path': 'output_ISIC_2_small_clusters_f=0.1'
}
data['ISIC_2_small_clusters_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_small_clusters_f=0.2',
'output_path': 'output_ISIC_2_small_clusters_f=0.2'
}
data['ISIC_2_small_clusters_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_small_clusters_f=0.3',
'output_path': 'output_ISIC_2_small_clusters_f=0.3'
}
data['ISIC_2_small_clusters_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_small_clusters_f=0.4',
'output_path': 'output_ISIC_2_small_clusters_f=0.4'
}
data['ISIC_2_small_clusters_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_small_clusters_f=0.5',
'output_path': 'output_ISIC_2_small_clusters_f=0.5'
}
data['ISIC_2_small_clusters_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_small_clusters_f=0.6',
'output_path': 'output_ISIC_2_small_clusters_f=0.6'
}
data['ISIC_2_small_clusters_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_small_clusters_f=0.7',
'output_path': 'output_ISIC_2_small_clusters_f=0.7'
}
data['ISIC_2_small_clusters_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_small_clusters_f=0.8',
'output_path': 'output_ISIC_2_small_clusters_f=0.8'
}
data['ISIC_2_small_clusters_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_small_clusters_f=0.9',
'output_path': 'output_ISIC_2_small_clusters_f=0.9'
}
data['ISIC_2_small_clusters_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_2',
'dataset_path': 'dataset_ISIC_2_small_clusters_f=1.0',
'output_path': 'output_ISIC_2_small_clusters_f=1.0'
}

data['ISIC_3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3',
  'output_path': 'output_ISIC_3'
  }
data['ISIC_3_image_rot_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_image_rot_f=0.1',
  'output_path': 'output_ISIC_3_image_rot_f=0.1'
  }
data['ISIC_3_image_rot_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_image_rot_f=0.2',
  'output_path': 'output_ISIC_3_image_rot_f=0.2'
  }
data['ISIC_3_image_rot_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_image_rot_f=0.3',
  'output_path': 'output_ISIC_3_image_rot_f=0.3'
  }
data['ISIC_3_image_rot_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_image_rot_f=0.4',
  'output_path': 'output_ISIC_3_image_rot_f=0.4'
  }
data['ISIC_3_image_rot_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_image_rot_f=0.5',
  'output_path': 'output_ISIC_3_image_rot_f=0.5'
  }
data['ISIC_3_image_rot_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_image_rot_f=0.6',
  'output_path': 'output_ISIC_3_image_rot_f=0.6'
  }
data['ISIC_3_image_rot_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_image_rot_f=0.7',
  'output_path': 'output_ISIC_3_image_rot_f=0.7'
  }
data['ISIC_3_image_rot_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_image_rot_f=0.8',
  'output_path': 'output_ISIC_3_image_rot_f=0.8'
  }
data['ISIC_3_image_rot_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_image_rot_f=0.9',
  'output_path': 'output_ISIC_3_image_rot_f=0.9'
  }
data['ISIC_3_image_rot_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_image_rot_f=1.0',
  'output_path': 'output_ISIC_3_image_rot_f=1.0'
  }

data['ISIC_3_image_translation_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_image_translation_f=0.1',
  'output_path': 'output_ISIC_3_image_translation_f=0.1'
  }
data['ISIC_3_image_translation_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_image_translation_f=0.2',
  'output_path': 'output_ISIC_3_image_translation_f=0.2'
  }
data['ISIC_3_image_translation_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_image_translation_f=0.3',
  'output_path': 'output_ISIC_3_image_translation_f=0.3'
  }
data['ISIC_3_image_translation_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_image_translation_f=0.4',
  'output_path': 'output_ISIC_3_image_translation_f=0.4'
  }
data['ISIC_3_image_translation_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_image_translation_f=0.5',
  'output_path': 'output_ISIC_3_image_translation_f=0.5'
  }
data['ISIC_3_image_translation_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_image_translation_f=0.6',
  'output_path': 'output_ISIC_3_image_translation_f=0.6'
  }
data['ISIC_3_image_translation_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_image_translation_f=0.7',
  'output_path': 'output_ISIC_3_image_translation_f=0.7'
  }
data['ISIC_3_image_translation_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_image_translation_f=0.8',
  'output_path': 'output_ISIC_3_image_translation_f=0.8'
  }
data['ISIC_3_image_translation_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_image_translation_f=0.9',
  'output_path': 'output_ISIC_3_image_translation_f=0.9'
  }
data['ISIC_3_image_translation_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_image_translation_f=1.0',
  'output_path': 'output_ISIC_3_image_translation_f=1.0'
  }

data['ISIC_3_image_zoom_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_image_zoom_f=0.1',
  'output_path': 'output_ISIC_3_image_zoom_f=0.1'
  }
data['ISIC_3_image_zoom_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_image_zoom_f=0.2',
  'output_path': 'output_ISIC_3_image_zoom_f=0.2'
  }
data['ISIC_3_image_zoom_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_image_zoom_f=0.3',
  'output_path': 'output_ISIC_3_image_zoom_f=0.3'
  }
data['ISIC_3_image_zoom_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_image_zoom_f=0.4',
  'output_path': 'output_ISIC_3_image_zoom_f=0.4'
  }
data['ISIC_3_image_zoom_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_image_zoom_f=0.5',
  'output_path': 'output_ISIC_3_image_zoom_f=0.5'
  }
data['ISIC_3_image_zoom_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_image_zoom_f=0.6',
  'output_path': 'output_ISIC_3_image_zoom_f=0.6'
  }
data['ISIC_3_image_zoom_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_image_zoom_f=0.7',
  'output_path': 'output_ISIC_3_image_zoom_f=0.7'
  }
data['ISIC_3_image_zoom_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_image_zoom_f=0.8',
  'output_path': 'output_ISIC_3_image_zoom_f=0.8'
  }
data['ISIC_3_image_zoom_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_image_zoom_f=0.9',
  'output_path': 'output_ISIC_3_image_zoom_f=0.9'
  }
data['ISIC_3_image_zoom_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_image_zoom_f=1.0',
  'output_path': 'output_ISIC_3_image_zoom_f=1.0'
  }

data['ISIC_3_add_noise_gaussian_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_gaussian_f=0.1',
  'output_path': 'output_ISIC_3_add_noise_gaussian_f=0.1'
  }
data['ISIC_3_add_noise_gaussian_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_gaussian_f=0.2',
  'output_path': 'output_ISIC_3_add_noise_gaussian_f=0.2'
  }
data['ISIC_3_add_noise_gaussian_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_gaussian_f=0.3',
  'output_path': 'output_ISIC_3_add_noise_gaussian_f=0.3'
  }
data['ISIC_3_add_noise_gaussian_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_gaussian_f=0.4',
  'output_path': 'output_ISIC_3_add_noise_gaussian_f=0.4'
  }
data['ISIC_3_add_noise_gaussian_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_gaussian_f=0.5',
  'output_path': 'output_ISIC_3_add_noise_gaussian_f=0.5'
  }
data['ISIC_3_add_noise_gaussian_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_gaussian_f=0.6',
  'output_path': 'output_ISIC_3_add_noise_gaussian_f=0.6'
  }
data['ISIC_3_add_noise_gaussian_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_gaussian_f=0.7',
  'output_path': 'output_ISIC_3_add_noise_gaussian_f=0.7'
  }
data['ISIC_3_add_noise_gaussian_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_gaussian_f=0.8',
  'output_path': 'output_ISIC_3_add_noise_gaussian_f=0.8'
  }
data['ISIC_3_add_noise_gaussian_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_gaussian_f=0.9',
  'output_path': 'output_ISIC_3_add_noise_gaussian_f=0.9'
  }
data['ISIC_3_add_noise_gaussian_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_gaussian_f=1.0',
  'output_path': 'output_ISIC_3_add_noise_gaussian_f=1.0'
  }

data['ISIC_3_add_noise_poisson_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_poisson_f=0.1',
  'output_path': 'output_ISIC_3_add_noise_poisson_f=0.1'
  }
data['ISIC_3_add_noise_poisson_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_poisson_f=0.2',
  'output_path': 'output_ISIC_3_add_noise_poisson_f=0.2'
  }
data['ISIC_3_add_noise_poisson_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_poisson_f=0.3',
  'output_path': 'output_ISIC_3_add_noise_poisson_f=0.3'
  }
data['ISIC_3_add_noise_poisson_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_poisson_f=0.4',
  'output_path': 'output_ISIC_3_add_noise_poisson_f=0.4'
  }
data['ISIC_3_add_noise_poisson_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_poisson_f=0.5',
  'output_path': 'output_ISIC_3_add_noise_poisson_f=0.5'
  }
data['ISIC_3_add_noise_poisson_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_poisson_f=0.6',
  'output_path': 'output_ISIC_3_add_noise_poisson_f=0.6'
  }
data['ISIC_3_add_noise_poisson_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_poisson_f=0.7',
  'output_path': 'output_ISIC_3_add_noise_poisson_f=0.7'
  }
data['ISIC_3_add_noise_poisson_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_poisson_f=0.8',
  'output_path': 'output_ISIC_3_add_noise_poisson_f=0.8'
  }
data['ISIC_3_add_noise_poisson_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_poisson_f=0.9',
  'output_path': 'output_ISIC_3_add_noise_poisson_f=0.9'
  }
data['ISIC_3_add_noise_poisson_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_poisson_f=1.0',
  'output_path': 'output_ISIC_3_add_noise_poisson_f=1.0'
  }

data['ISIC_3_add_noise_salt_and_pepper_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_salt_and_pepper_f=0.1',
  'output_path': 'output_ISIC_3_add_noise_salt_and_pepper_f=0.1'
  }
data['ISIC_3_add_noise_salt_and_pepper_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_salt_and_pepper_f=0.2',
  'output_path': 'output_ISIC_3_add_noise_salt_and_pepper_f=0.2'
  }
data['ISIC_3_add_noise_salt_and_pepper_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_salt_and_pepper_f=0.3',
  'output_path': 'output_ISIC_3_add_noise_salt_and_pepper_f=0.3'
  }
data['ISIC_3_add_noise_salt_and_pepper_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_salt_and_pepper_f=0.4',
  'output_path': 'output_ISIC_3_add_noise_salt_and_pepper_f=0.4'
  }
data['ISIC_3_add_noise_salt_and_pepper_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_salt_and_pepper_f=0.5',
  'output_path': 'output_ISIC_3_add_noise_salt_and_pepper_f=0.5'
  }
data['ISIC_3_add_noise_salt_and_pepper_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_salt_and_pepper_f=0.6',
  'output_path': 'output_ISIC_3_add_noise_salt_and_pepper_f=0.6'
  }
data['ISIC_3_add_noise_salt_and_pepper_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_salt_and_pepper_f=0.7',
  'output_path': 'output_ISIC_3_add_noise_salt_and_pepper_f=0.7'
  }
data['ISIC_3_add_noise_salt_and_pepper_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_salt_and_pepper_f=0.8',
  'output_path': 'output_ISIC_3_add_noise_salt_and_pepper_f=0.8'
  }
data['ISIC_3_add_noise_salt_and_pepper_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_salt_and_pepper_f=0.9',
  'output_path': 'output_ISIC_3_add_noise_salt_and_pepper_f=0.9'
  }
data['ISIC_3_add_noise_salt_and_pepper_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_salt_and_pepper_f=1.0',
  'output_path': 'output_ISIC_3_add_noise_salt_and_pepper_f=1.0'
  }

data['ISIC_3_add_noise_speckle_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_speckle_f=0.1',
  'output_path': 'output_ISIC_3_add_noise_speckle_f=0.1'
  }
data['ISIC_3_add_noise_speckle_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_speckle_f=0.2',
  'output_path': 'output_ISIC_3_add_noise_speckle_f=0.2'
  }
data['ISIC_3_add_noise_speckle_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_speckle_f=0.3',
  'output_path': 'output_ISIC_3_add_noise_speckle_f=0.3'
  }
data['ISIC_3_add_noise_speckle_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_speckle_f=0.4',
  'output_path': 'output_ISIC_3_add_noise_speckle_f=0.4'
  }
data['ISIC_3_add_noise_speckle_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_speckle_f=0.5',
  'output_path': 'output_ISIC_3_add_noise_speckle_f=0.5'
  }
data['ISIC_3_add_noise_speckle_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_speckle_f=0.6',
  'output_path': 'output_ISIC_3_add_noise_speckle_f=0.6'
  }
data['ISIC_3_add_noise_speckle_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_speckle_f=0.7',
  'output_path': 'output_ISIC_3_add_noise_speckle_f=0.7'
  }
data['ISIC_3_add_noise_speckle_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_speckle_f=0.8',
  'output_path': 'output_ISIC_3_add_noise_speckle_f=0.8'
  }
data['ISIC_3_add_noise_speckle_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_speckle_f=0.9',
  'output_path': 'output_ISIC_3_add_noise_speckle_f=0.9'
  }
data['ISIC_3_add_noise_speckle_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_add_noise_speckle_f=1.0',
  'output_path': 'output_ISIC_3_add_noise_speckle_f=1.0'
  }

data['ISIC_3_imbalance_classes_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_imbalance_classes_f=0.1',
'output_path': 'output_ISIC_3_imbalance_classes_f=0.1'
}
data['ISIC_3_imbalance_classes_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_imbalance_classes_f=0.2',
'output_path': 'output_ISIC_3_imbalance_classes_f=0.2'
}
data['ISIC_3_imbalance_classes_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_imbalance_classes_f=0.3',
'output_path': 'output_ISIC_3_imbalance_classes_f=0.3'
}
data['ISIC_3_imbalance_classes_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_imbalance_classes_f=0.4',
'output_path': 'output_ISIC_3_imbalance_classes_f=0.4'
}
data['ISIC_3_imbalance_classes_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_imbalance_classes_f=0.5',
'output_path': 'output_ISIC_3_imbalance_classes_f=0.5'
}
data['ISIC_3_imbalance_classes_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_imbalance_classes_f=0.6',
'output_path': 'output_ISIC_3_imbalance_classes_f=0.6'
}
data['ISIC_3_imbalance_classes_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_imbalance_classes_f=0.7',
'output_path': 'output_ISIC_3_imbalance_classes_f=0.7'
}
data['ISIC_3_imbalance_classes_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_imbalance_classes_f=0.8',
'output_path': 'output_ISIC_3_imbalance_classes_f=0.8'
}
data['ISIC_3_imbalance_classes_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_imbalance_classes_f=0.9',
'output_path': 'output_ISIC_3_imbalance_classes_f=0.9'
}
data['ISIC_3_imbalance_classes_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_imbalance_classes_f=1.0',
'output_path': 'output_ISIC_3_imbalance_classes_f=1.0'
}

data['ISIC_3_grayscale_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_grayscale_f=0.1',
'output_path': 'output_ISIC_3_grayscale_f=0.1'
}
data['ISIC_3_grayscale_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_grayscale_f=0.2',
'output_path': 'output_ISIC_3_grayscale_f=0.2'
}
data['ISIC_3_grayscale_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_grayscale_f=0.3',
'output_path': 'output_ISIC_3_grayscale_f=0.3'
}
data['ISIC_3_grayscale_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_grayscale_f=0.4',
'output_path': 'output_ISIC_3_grayscale_f=0.4'
}
data['ISIC_3_grayscale_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_grayscale_f=0.5',
'output_path': 'output_ISIC_3_grayscale_f=0.5'
}
data['ISIC_3_grayscale_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_grayscale_f=0.6',
'output_path': 'output_ISIC_3_grayscale_f=0.6'
}
data['ISIC_3_grayscale_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_grayscale_f=0.7',
'output_path': 'output_ISIC_3_grayscale_f=0.7'
}
data['ISIC_3_grayscale_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_grayscale_f=0.8',
'output_path': 'output_ISIC_3_grayscale_f=0.8'
}
data['ISIC_3_grayscale_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_grayscale_f=0.9',
'output_path': 'output_ISIC_3_grayscale_f=0.9'
}
data['ISIC_3_grayscale_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_grayscale_f=1.0',
'output_path': 'output_ISIC_3_grayscale_f=1.0'
}

data['ISIC_3_hsv_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_hsv_f=0.1',
'output_path': 'output_ISIC_3_hsv_f=0.1'
}
data['ISIC_3_hsv_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_hsv_f=0.2',
'output_path': 'output_ISIC_3_hsv_f=0.2'
}
data['ISIC_3_hsv_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_hsv_f=0.3',
'output_path': 'output_ISIC_3_hsv_f=0.3'
}
data['ISIC_3_hsv_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_hsv_f=0.4',
'output_path': 'output_ISIC_3_hsv_f=0.4'
}
data['ISIC_3_hsv_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_hsv_f=0.5',
'output_path': 'output_ISIC_3_hsv_f=0.5'
}
data['ISIC_3_hsv_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_hsv_f=0.6',
'output_path': 'output_ISIC_3_hsv_f=0.6'
}
data['ISIC_3_hsv_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_hsv_f=0.7',
'output_path': 'output_ISIC_3_hsv_f=0.7'
}
data['ISIC_3_hsv_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_hsv_f=0.8',
'output_path': 'output_ISIC_3_hsv_f=0.8'
}
data['ISIC_3_hsv_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_hsv_f=0.9',
'output_path': 'output_ISIC_3_hsv_f=0.9'
}
data['ISIC_3_hsv_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_hsv_f=1.0',
'output_path': 'output_ISIC_3_hsv_f=1.0'
}
data['ISIC_3_blur_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_blur_f=0.1',
'output_path': 'output_ISIC_3_blur_f=0.1'
}
data['ISIC_3_blur_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_blur_f=0.2',
'output_path': 'output_ISIC_3_blur_f=0.2'
}
data['ISIC_3_blur_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_blur_f=0.3',
'output_path': 'output_ISIC_3_blur_f=0.3'
}
data['ISIC_3_blur_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_blur_f=0.4',
'output_path': 'output_ISIC_3_blur_f=0.4'
}
data['ISIC_3_blur_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_blur_f=0.5',
'output_path': 'output_ISIC_3_blur_f=0.5'
}
data['ISIC_3_blur_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_blur_f=0.6',
'output_path': 'output_ISIC_3_blur_f=0.6'
}
data['ISIC_3_blur_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_blur_f=0.7',
'output_path': 'output_ISIC_3_blur_f=0.7'
}
data['ISIC_3_blur_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_blur_f=0.8',
'output_path': 'output_ISIC_3_blur_f=0.8'
}
data['ISIC_3_blur_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_blur_f=0.9',
'output_path': 'output_ISIC_3_blur_f=0.9'
}
data['ISIC_3_blur_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_blur_f=1.0',
'output_path': 'output_ISIC_3_blur_f=1.0'
}

data['ISIC_3_small_random_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_small_random_f=0.1',
'output_path': 'output_ISIC_3_small_random_f=0.1'
}
data['ISIC_3_small_random_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_small_random_f=0.2',
'output_path': 'output_ISIC_3_small_random_f=0.2'
}
data['ISIC_3_small_random_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_small_random_f=0.3',
'output_path': 'output_ISIC_3_small_random_f=0.3'
}
data['ISIC_3_small_random_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_small_random_f=0.4',
'output_path': 'output_ISIC_3_small_random_f=0.4'
}
data['ISIC_3_small_random_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_small_random_f=0.5',
'output_path': 'output_ISIC_3_small_random_f=0.5'
}
data['ISIC_3_small_random_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_small_random_f=0.6',
'output_path': 'output_ISIC_3_small_random_f=0.6'
}
data['ISIC_3_small_random_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_small_random_f=0.7',
'output_path': 'output_ISIC_3_small_random_f=0.7'
}
data['ISIC_3_small_random_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_small_random_f=0.8',
'output_path': 'output_ISIC_3_small_random_f=0.8'
}
data['ISIC_3_small_random_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_small_random_f=0.9',
'output_path': 'output_ISIC_3_small_random_f=0.9'
}
data['ISIC_3_small_random_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_small_random_f=1.0',
'output_path': 'output_ISIC_3_small_random_f=1.0'
}

data['ISIC_3_small_easy_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_small_easy_f=0.1',
  'output_path': 'output_ISIC_3_small_easy_f=0.1'
  }
data['ISIC_3_small_easy_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_small_easy_f=0.2',
  'output_path': 'output_ISIC_3_small_easy_f=0.2'
  }
data['ISIC_3_small_easy_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_small_easy_f=0.3',
  'output_path': 'output_ISIC_3_small_easy_f=0.3'
  }
data['ISIC_3_small_easy_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_small_easy_f=0.4',
  'output_path': 'output_ISIC_3_small_easy_f=0.4'
  }
data['ISIC_3_small_easy_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_small_easy_f=0.5',
  'output_path': 'output_ISIC_3_small_easy_f=0.5'
  }
data['ISIC_3_small_easy_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_small_easy_f=0.6',
  'output_path': 'output_ISIC_3_small_easy_f=0.6'
  }
data['ISIC_3_small_easy_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_small_easy_f=0.7',
  'output_path': 'output_ISIC_3_small_easy_f=0.7'
  }
data['ISIC_3_small_easy_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_small_easy_f=0.8',
  'output_path': 'output_ISIC_3_small_easy_f=0.8'
  }
data['ISIC_3_small_easy_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_small_easy_f=0.9',
  'output_path': 'output_ISIC_3_small_easy_f=0.9'
  }
data['ISIC_3_small_easy_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_small_easy_f=1.0',
  'output_path': 'output_ISIC_3_small_easy_f=1.0'
  }

data['ISIC_3_small_hard_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_small_hard_f=0.1',
  'output_path': 'output_ISIC_3_small_hard_f=0.1'
  }
data['ISIC_3_small_hard_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_small_hard_f=0.2',
  'output_path': 'output_ISIC_3_small_hard_f=0.2'
  }
data['ISIC_3_small_hard_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_small_hard_f=0.3',
  'output_path': 'output_ISIC_3_small_hard_f=0.3'
  }
data['ISIC_3_small_hard_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_small_hard_f=0.4',
  'output_path': 'output_ISIC_3_small_hard_f=0.4'
  }
data['ISIC_3_small_hard_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_small_hard_f=0.5',
  'output_path': 'output_ISIC_3_small_hard_f=0.5'
  }
data['ISIC_3_small_hard_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_small_hard_f=0.6',
  'output_path': 'output_ISIC_3_small_hard_f=0.6'
  }
data['ISIC_3_small_hard_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_small_hard_f=0.7',
  'output_path': 'output_ISIC_3_small_hard_f=0.7'
  }
data['ISIC_3_small_hard_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_small_hard_f=0.8',
  'output_path': 'output_ISIC_3_small_hard_f=0.8'
  }
data['ISIC_3_small_hard_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_small_hard_f=0.9',
  'output_path': 'output_ISIC_3_small_hard_f=0.9'
  }
data['ISIC_3_small_hard_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_3',
  'dataset_path': 'dataset_ISIC_3_small_hard_f=1.0',
  'output_path': 'output_ISIC_3_small_hard_f=1.0'
  }

data['ISIC_3_small_clusters_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_small_clusters_f=0.1',
'output_path': 'output_ISIC_3_small_clusters_f=0.1'
}
data['ISIC_3_small_clusters_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_small_clusters_f=0.2',
'output_path': 'output_ISIC_3_small_clusters_f=0.2'
}
data['ISIC_3_small_clusters_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_small_clusters_f=0.3',
'output_path': 'output_ISIC_3_small_clusters_f=0.3'
}
data['ISIC_3_small_clusters_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_small_clusters_f=0.4',
'output_path': 'output_ISIC_3_small_clusters_f=0.4'
}
data['ISIC_3_small_clusters_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_small_clusters_f=0.5',
'output_path': 'output_ISIC_3_small_clusters_f=0.5'
}
data['ISIC_3_small_clusters_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_small_clusters_f=0.6',
'output_path': 'output_ISIC_3_small_clusters_f=0.6'
}
data['ISIC_3_small_clusters_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_small_clusters_f=0.7',
'output_path': 'output_ISIC_3_small_clusters_f=0.7'
}
data['ISIC_3_small_clusters_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_small_clusters_f=0.8',
'output_path': 'output_ISIC_3_small_clusters_f=0.8'
}
data['ISIC_3_small_clusters_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_small_clusters_f=0.9',
'output_path': 'output_ISIC_3_small_clusters_f=0.9'
}
data['ISIC_3_small_clusters_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_3',
'dataset_path': 'dataset_ISIC_3_small_clusters_f=1.0',
'output_path': 'output_ISIC_3_small_clusters_f=1.0'
}

data['ISIC_4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4',
  'output_path': 'output_ISIC_4'
  }
data['ISIC_4_image_rot_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_image_rot_f=0.1',
  'output_path': 'output_ISIC_4_image_rot_f=0.1'
  }
data['ISIC_4_image_rot_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_image_rot_f=0.2',
  'output_path': 'output_ISIC_4_image_rot_f=0.2'
  }
data['ISIC_4_image_rot_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_image_rot_f=0.3',
  'output_path': 'output_ISIC_4_image_rot_f=0.3'
  }
data['ISIC_4_image_rot_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_image_rot_f=0.4',
  'output_path': 'output_ISIC_4_image_rot_f=0.4'
  }
data['ISIC_4_image_rot_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_image_rot_f=0.5',
  'output_path': 'output_ISIC_4_image_rot_f=0.5'
  }
data['ISIC_4_image_rot_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_image_rot_f=0.6',
  'output_path': 'output_ISIC_4_image_rot_f=0.6'
  }
data['ISIC_4_image_rot_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_image_rot_f=0.7',
  'output_path': 'output_ISIC_4_image_rot_f=0.7'
  }
data['ISIC_4_image_rot_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_image_rot_f=0.8',
  'output_path': 'output_ISIC_4_image_rot_f=0.8'
  }
data['ISIC_4_image_rot_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_image_rot_f=0.9',
  'output_path': 'output_ISIC_4_image_rot_f=0.9'
  }
data['ISIC_4_image_rot_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_image_rot_f=1.0',
  'output_path': 'output_ISIC_4_image_rot_f=1.0'
  }

data['ISIC_4_image_translation_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_image_translation_f=0.1',
  'output_path': 'output_ISIC_4_image_translation_f=0.1'
  }
data['ISIC_4_image_translation_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_image_translation_f=0.2',
  'output_path': 'output_ISIC_4_image_translation_f=0.2'
  }
data['ISIC_4_image_translation_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_image_translation_f=0.3',
  'output_path': 'output_ISIC_4_image_translation_f=0.3'
  }
data['ISIC_4_image_translation_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_image_translation_f=0.4',
  'output_path': 'output_ISIC_4_image_translation_f=0.4'
  }
data['ISIC_4_image_translation_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_image_translation_f=0.5',
  'output_path': 'output_ISIC_4_image_translation_f=0.5'
  }
data['ISIC_4_image_translation_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_image_translation_f=0.6',
  'output_path': 'output_ISIC_4_image_translation_f=0.6'
  }
data['ISIC_4_image_translation_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_image_translation_f=0.7',
  'output_path': 'output_ISIC_4_image_translation_f=0.7'
  }
data['ISIC_4_image_translation_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_image_translation_f=0.8',
  'output_path': 'output_ISIC_4_image_translation_f=0.8'
  }
data['ISIC_4_image_translation_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_image_translation_f=0.9',
  'output_path': 'output_ISIC_4_image_translation_f=0.9'
  }
data['ISIC_4_image_translation_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_image_translation_f=1.0',
  'output_path': 'output_ISIC_4_image_translation_f=1.0'
  }

data['ISIC_4_image_zoom_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_image_zoom_f=0.1',
  'output_path': 'output_ISIC_4_image_zoom_f=0.1'
  }
data['ISIC_4_image_zoom_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_image_zoom_f=0.2',
  'output_path': 'output_ISIC_4_image_zoom_f=0.2'
  }
data['ISIC_4_image_zoom_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_image_zoom_f=0.3',
  'output_path': 'output_ISIC_4_image_zoom_f=0.3'
  }
data['ISIC_4_image_zoom_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_image_zoom_f=0.4',
  'output_path': 'output_ISIC_4_image_zoom_f=0.4'
  }
data['ISIC_4_image_zoom_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_image_zoom_f=0.5',
  'output_path': 'output_ISIC_4_image_zoom_f=0.5'
  }
data['ISIC_4_image_zoom_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_image_zoom_f=0.6',
  'output_path': 'output_ISIC_4_image_zoom_f=0.6'
  }
data['ISIC_4_image_zoom_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_image_zoom_f=0.7',
  'output_path': 'output_ISIC_4_image_zoom_f=0.7'
  }
data['ISIC_4_image_zoom_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_image_zoom_f=0.8',
  'output_path': 'output_ISIC_4_image_zoom_f=0.8'
  }
data['ISIC_4_image_zoom_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_image_zoom_f=0.9',
  'output_path': 'output_ISIC_4_image_zoom_f=0.9'
  }
data['ISIC_4_image_zoom_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_image_zoom_f=1.0',
  'output_path': 'output_ISIC_4_image_zoom_f=1.0'
  }

data['ISIC_4_add_noise_gaussian_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_gaussian_f=0.1',
  'output_path': 'output_ISIC_4_add_noise_gaussian_f=0.1'
  }
data['ISIC_4_add_noise_gaussian_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_gaussian_f=0.2',
  'output_path': 'output_ISIC_4_add_noise_gaussian_f=0.2'
  }
data['ISIC_4_add_noise_gaussian_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_gaussian_f=0.3',
  'output_path': 'output_ISIC_4_add_noise_gaussian_f=0.3'
  }
data['ISIC_4_add_noise_gaussian_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_gaussian_f=0.4',
  'output_path': 'output_ISIC_4_add_noise_gaussian_f=0.4'
  }
data['ISIC_4_add_noise_gaussian_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_gaussian_f=0.5',
  'output_path': 'output_ISIC_4_add_noise_gaussian_f=0.5'
  }
data['ISIC_4_add_noise_gaussian_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_gaussian_f=0.6',
  'output_path': 'output_ISIC_4_add_noise_gaussian_f=0.6'
  }
data['ISIC_4_add_noise_gaussian_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_gaussian_f=0.7',
  'output_path': 'output_ISIC_4_add_noise_gaussian_f=0.7'
  }
data['ISIC_4_add_noise_gaussian_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_gaussian_f=0.8',
  'output_path': 'output_ISIC_4_add_noise_gaussian_f=0.8'
  }
data['ISIC_4_add_noise_gaussian_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_gaussian_f=0.9',
  'output_path': 'output_ISIC_4_add_noise_gaussian_f=0.9'
  }
data['ISIC_4_add_noise_gaussian_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_gaussian_f=1.0',
  'output_path': 'output_ISIC_4_add_noise_gaussian_f=1.0'
  }

data['ISIC_4_add_noise_poisson_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_poisson_f=0.1',
  'output_path': 'output_ISIC_4_add_noise_poisson_f=0.1'
  }
data['ISIC_4_add_noise_poisson_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_poisson_f=0.2',
  'output_path': 'output_ISIC_4_add_noise_poisson_f=0.2'
  }
data['ISIC_4_add_noise_poisson_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_poisson_f=0.3',
  'output_path': 'output_ISIC_4_add_noise_poisson_f=0.3'
  }
data['ISIC_4_add_noise_poisson_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_poisson_f=0.4',
  'output_path': 'output_ISIC_4_add_noise_poisson_f=0.4'
  }
data['ISIC_4_add_noise_poisson_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_poisson_f=0.5',
  'output_path': 'output_ISIC_4_add_noise_poisson_f=0.5'
  }
data['ISIC_4_add_noise_poisson_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_poisson_f=0.6',
  'output_path': 'output_ISIC_4_add_noise_poisson_f=0.6'
  }
data['ISIC_4_add_noise_poisson_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_poisson_f=0.7',
  'output_path': 'output_ISIC_4_add_noise_poisson_f=0.7'
  }
data['ISIC_4_add_noise_poisson_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_poisson_f=0.8',
  'output_path': 'output_ISIC_4_add_noise_poisson_f=0.8'
  }
data['ISIC_4_add_noise_poisson_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_poisson_f=0.9',
  'output_path': 'output_ISIC_4_add_noise_poisson_f=0.9'
  }
data['ISIC_4_add_noise_poisson_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_poisson_f=1.0',
  'output_path': 'output_ISIC_4_add_noise_poisson_f=1.0'
  }

data['ISIC_4_add_noise_salt_and_pepper_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_salt_and_pepper_f=0.1',
  'output_path': 'output_ISIC_4_add_noise_salt_and_pepper_f=0.1'
  }
data['ISIC_4_add_noise_salt_and_pepper_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_salt_and_pepper_f=0.2',
  'output_path': 'output_ISIC_4_add_noise_salt_and_pepper_f=0.2'
  }
data['ISIC_4_add_noise_salt_and_pepper_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_salt_and_pepper_f=0.3',
  'output_path': 'output_ISIC_4_add_noise_salt_and_pepper_f=0.3'
  }
data['ISIC_4_add_noise_salt_and_pepper_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_salt_and_pepper_f=0.4',
  'output_path': 'output_ISIC_4_add_noise_salt_and_pepper_f=0.4'
  }
data['ISIC_4_add_noise_salt_and_pepper_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_salt_and_pepper_f=0.5',
  'output_path': 'output_ISIC_4_add_noise_salt_and_pepper_f=0.5'
  }
data['ISIC_4_add_noise_salt_and_pepper_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_salt_and_pepper_f=0.6',
  'output_path': 'output_ISIC_4_add_noise_salt_and_pepper_f=0.6'
  }
data['ISIC_4_add_noise_salt_and_pepper_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_salt_and_pepper_f=0.7',
  'output_path': 'output_ISIC_4_add_noise_salt_and_pepper_f=0.7'
  }
data['ISIC_4_add_noise_salt_and_pepper_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_salt_and_pepper_f=0.8',
  'output_path': 'output_ISIC_4_add_noise_salt_and_pepper_f=0.8'
  }
data['ISIC_4_add_noise_salt_and_pepper_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_salt_and_pepper_f=0.9',
  'output_path': 'output_ISIC_4_add_noise_salt_and_pepper_f=0.9'
  }
data['ISIC_4_add_noise_salt_and_pepper_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_salt_and_pepper_f=1.0',
  'output_path': 'output_ISIC_4_add_noise_salt_and_pepper_f=1.0'
  }

data['ISIC_4_add_noise_speckle_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_speckle_f=0.1',
  'output_path': 'output_ISIC_4_add_noise_speckle_f=0.1'
  }
data['ISIC_4_add_noise_speckle_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_speckle_f=0.2',
  'output_path': 'output_ISIC_4_add_noise_speckle_f=0.2'
  }
data['ISIC_4_add_noise_speckle_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_speckle_f=0.3',
  'output_path': 'output_ISIC_4_add_noise_speckle_f=0.3'
  }
data['ISIC_4_add_noise_speckle_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_speckle_f=0.4',
  'output_path': 'output_ISIC_4_add_noise_speckle_f=0.4'
  }
data['ISIC_4_add_noise_speckle_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_speckle_f=0.5',
  'output_path': 'output_ISIC_4_add_noise_speckle_f=0.5'
  }
data['ISIC_4_add_noise_speckle_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_speckle_f=0.6',
  'output_path': 'output_ISIC_4_add_noise_speckle_f=0.6'
  }
data['ISIC_4_add_noise_speckle_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_speckle_f=0.7',
  'output_path': 'output_ISIC_4_add_noise_speckle_f=0.7'
  }
data['ISIC_4_add_noise_speckle_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_speckle_f=0.8',
  'output_path': 'output_ISIC_4_add_noise_speckle_f=0.8'
  }
data['ISIC_4_add_noise_speckle_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_speckle_f=0.9',
  'output_path': 'output_ISIC_4_add_noise_speckle_f=0.9'
  }
data['ISIC_4_add_noise_speckle_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_add_noise_speckle_f=1.0',
  'output_path': 'output_ISIC_4_add_noise_speckle_f=1.0'
  }

data['ISIC_4_imbalance_classes_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_imbalance_classes_f=0.1',
'output_path': 'output_ISIC_4_imbalance_classes_f=0.1'
}
data['ISIC_4_imbalance_classes_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_imbalance_classes_f=0.2',
'output_path': 'output_ISIC_4_imbalance_classes_f=0.2'
}
data['ISIC_4_imbalance_classes_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_imbalance_classes_f=0.3',
'output_path': 'output_ISIC_4_imbalance_classes_f=0.3'
}
data['ISIC_4_imbalance_classes_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_imbalance_classes_f=0.4',
'output_path': 'output_ISIC_4_imbalance_classes_f=0.4'
}
data['ISIC_4_imbalance_classes_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_imbalance_classes_f=0.5',
'output_path': 'output_ISIC_4_imbalance_classes_f=0.5'
}
data['ISIC_4_imbalance_classes_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_imbalance_classes_f=0.6',
'output_path': 'output_ISIC_4_imbalance_classes_f=0.6'
}
data['ISIC_4_imbalance_classes_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_imbalance_classes_f=0.7',
'output_path': 'output_ISIC_4_imbalance_classes_f=0.7'
}
data['ISIC_4_imbalance_classes_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_imbalance_classes_f=0.8',
'output_path': 'output_ISIC_4_imbalance_classes_f=0.8'
}
data['ISIC_4_imbalance_classes_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_imbalance_classes_f=0.9',
'output_path': 'output_ISIC_4_imbalance_classes_f=0.9'
}
data['ISIC_4_imbalance_classes_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_imbalance_classes_f=1.0',
'output_path': 'output_ISIC_4_imbalance_classes_f=1.0'
}

data['ISIC_4_grayscale_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_grayscale_f=0.1',
'output_path': 'output_ISIC_4_grayscale_f=0.1'
}
data['ISIC_4_grayscale_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_grayscale_f=0.2',
'output_path': 'output_ISIC_4_grayscale_f=0.2'
}
data['ISIC_4_grayscale_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_grayscale_f=0.3',
'output_path': 'output_ISIC_4_grayscale_f=0.3'
}
data['ISIC_4_grayscale_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_grayscale_f=0.4',
'output_path': 'output_ISIC_4_grayscale_f=0.4'
}
data['ISIC_4_grayscale_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_grayscale_f=0.5',
'output_path': 'output_ISIC_4_grayscale_f=0.5'
}
data['ISIC_4_grayscale_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_grayscale_f=0.6',
'output_path': 'output_ISIC_4_grayscale_f=0.6'
}
data['ISIC_4_grayscale_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_grayscale_f=0.7',
'output_path': 'output_ISIC_4_grayscale_f=0.7'
}
data['ISIC_4_grayscale_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_grayscale_f=0.8',
'output_path': 'output_ISIC_4_grayscale_f=0.8'
}
data['ISIC_4_grayscale_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_grayscale_f=0.9',
'output_path': 'output_ISIC_4_grayscale_f=0.9'
}
data['ISIC_4_grayscale_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_grayscale_f=1.0',
'output_path': 'output_ISIC_4_grayscale_f=1.0'
}

data['ISIC_4_hsv_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_hsv_f=0.1',
'output_path': 'output_ISIC_4_hsv_f=0.1'
}
data['ISIC_4_hsv_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_hsv_f=0.2',
'output_path': 'output_ISIC_4_hsv_f=0.2'
}
data['ISIC_4_hsv_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_hsv_f=0.3',
'output_path': 'output_ISIC_4_hsv_f=0.3'
}
data['ISIC_4_hsv_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_hsv_f=0.4',
'output_path': 'output_ISIC_4_hsv_f=0.4'
}
data['ISIC_4_hsv_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_hsv_f=0.5',
'output_path': 'output_ISIC_4_hsv_f=0.5'
}
data['ISIC_4_hsv_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_hsv_f=0.6',
'output_path': 'output_ISIC_4_hsv_f=0.6'
}
data['ISIC_4_hsv_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_hsv_f=0.7',
'output_path': 'output_ISIC_4_hsv_f=0.7'
}
data['ISIC_4_hsv_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_hsv_f=0.8',
'output_path': 'output_ISIC_4_hsv_f=0.8'
}
data['ISIC_4_hsv_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_hsv_f=0.9',
'output_path': 'output_ISIC_4_hsv_f=0.9'
}
data['ISIC_4_hsv_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_hsv_f=1.0',
'output_path': 'output_ISIC_4_hsv_f=1.0'
}
data['ISIC_4_blur_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_blur_f=0.1',
'output_path': 'output_ISIC_4_blur_f=0.1'
}
data['ISIC_4_blur_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_blur_f=0.2',
'output_path': 'output_ISIC_4_blur_f=0.2'
}
data['ISIC_4_blur_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_blur_f=0.3',
'output_path': 'output_ISIC_4_blur_f=0.3'
}
data['ISIC_4_blur_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_blur_f=0.4',
'output_path': 'output_ISIC_4_blur_f=0.4'
}
data['ISIC_4_blur_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_blur_f=0.5',
'output_path': 'output_ISIC_4_blur_f=0.5'
}
data['ISIC_4_blur_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_blur_f=0.6',
'output_path': 'output_ISIC_4_blur_f=0.6'
}
data['ISIC_4_blur_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_blur_f=0.7',
'output_path': 'output_ISIC_4_blur_f=0.7'
}
data['ISIC_4_blur_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_blur_f=0.8',
'output_path': 'output_ISIC_4_blur_f=0.8'
}
data['ISIC_4_blur_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_blur_f=0.9',
'output_path': 'output_ISIC_4_blur_f=0.9'
}
data['ISIC_4_blur_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_blur_f=1.0',
'output_path': 'output_ISIC_4_blur_f=1.0'
}

data['ISIC_4_small_random_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_small_random_f=0.1',
'output_path': 'output_ISIC_4_small_random_f=0.1'
}
data['ISIC_4_small_random_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_small_random_f=0.2',
'output_path': 'output_ISIC_4_small_random_f=0.2'
}
data['ISIC_4_small_random_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_small_random_f=0.3',
'output_path': 'output_ISIC_4_small_random_f=0.3'
}
data['ISIC_4_small_random_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_small_random_f=0.4',
'output_path': 'output_ISIC_4_small_random_f=0.4'
}
data['ISIC_4_small_random_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_small_random_f=0.5',
'output_path': 'output_ISIC_4_small_random_f=0.5'
}
data['ISIC_4_small_random_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_small_random_f=0.6',
'output_path': 'output_ISIC_4_small_random_f=0.6'
}
data['ISIC_4_small_random_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_small_random_f=0.7',
'output_path': 'output_ISIC_4_small_random_f=0.7'
}
data['ISIC_4_small_random_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_small_random_f=0.8',
'output_path': 'output_ISIC_4_small_random_f=0.8'
}
data['ISIC_4_small_random_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_small_random_f=0.9',
'output_path': 'output_ISIC_4_small_random_f=0.9'
}
data['ISIC_4_small_random_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_small_random_f=1.0',
'output_path': 'output_ISIC_4_small_random_f=1.0'
}

data['ISIC_4_small_easy_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_small_easy_f=0.1',
  'output_path': 'output_ISIC_4_small_easy_f=0.1'
  }
data['ISIC_4_small_easy_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_small_easy_f=0.2',
  'output_path': 'output_ISIC_4_small_easy_f=0.2'
  }
data['ISIC_4_small_easy_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_small_easy_f=0.3',
  'output_path': 'output_ISIC_4_small_easy_f=0.3'
  }
data['ISIC_4_small_easy_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_small_easy_f=0.4',
  'output_path': 'output_ISIC_4_small_easy_f=0.4'
  }
data['ISIC_4_small_easy_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_small_easy_f=0.5',
  'output_path': 'output_ISIC_4_small_easy_f=0.5'
  }
data['ISIC_4_small_easy_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_small_easy_f=0.6',
  'output_path': 'output_ISIC_4_small_easy_f=0.6'
  }
data['ISIC_4_small_easy_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_small_easy_f=0.7',
  'output_path': 'output_ISIC_4_small_easy_f=0.7'
  }
data['ISIC_4_small_easy_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_small_easy_f=0.8',
  'output_path': 'output_ISIC_4_small_easy_f=0.8'
  }
data['ISIC_4_small_easy_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_small_easy_f=0.9',
  'output_path': 'output_ISIC_4_small_easy_f=0.9'
  }
data['ISIC_4_small_easy_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_small_easy_f=1.0',
  'output_path': 'output_ISIC_4_small_easy_f=1.0'
  }

data['ISIC_4_small_hard_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_small_hard_f=0.1',
  'output_path': 'output_ISIC_4_small_hard_f=0.1'
  }
data['ISIC_4_small_hard_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_small_hard_f=0.2',
  'output_path': 'output_ISIC_4_small_hard_f=0.2'
  }
data['ISIC_4_small_hard_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_small_hard_f=0.3',
  'output_path': 'output_ISIC_4_small_hard_f=0.3'
  }
data['ISIC_4_small_hard_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_small_hard_f=0.4',
  'output_path': 'output_ISIC_4_small_hard_f=0.4'
  }
data['ISIC_4_small_hard_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_small_hard_f=0.5',
  'output_path': 'output_ISIC_4_small_hard_f=0.5'
  }
data['ISIC_4_small_hard_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_small_hard_f=0.6',
  'output_path': 'output_ISIC_4_small_hard_f=0.6'
  }
data['ISIC_4_small_hard_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_small_hard_f=0.7',
  'output_path': 'output_ISIC_4_small_hard_f=0.7'
  }
data['ISIC_4_small_hard_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_small_hard_f=0.8',
  'output_path': 'output_ISIC_4_small_hard_f=0.8'
  }
data['ISIC_4_small_hard_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_small_hard_f=0.9',
  'output_path': 'output_ISIC_4_small_hard_f=0.9'
  }
data['ISIC_4_small_hard_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_4',
  'dataset_path': 'dataset_ISIC_4_small_hard_f=1.0',
  'output_path': 'output_ISIC_4_small_hard_f=1.0'
  }

data['ISIC_4_small_clusters_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_small_clusters_f=0.1',
'output_path': 'output_ISIC_4_small_clusters_f=0.1'
}
data['ISIC_4_small_clusters_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_small_clusters_f=0.2',
'output_path': 'output_ISIC_4_small_clusters_f=0.2'
}
data['ISIC_4_small_clusters_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_small_clusters_f=0.3',
'output_path': 'output_ISIC_4_small_clusters_f=0.3'
}
data['ISIC_4_small_clusters_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_small_clusters_f=0.4',
'output_path': 'output_ISIC_4_small_clusters_f=0.4'
}
data['ISIC_4_small_clusters_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_small_clusters_f=0.5',
'output_path': 'output_ISIC_4_small_clusters_f=0.5'
}
data['ISIC_4_small_clusters_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_small_clusters_f=0.6',
'output_path': 'output_ISIC_4_small_clusters_f=0.6'
}
data['ISIC_4_small_clusters_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_small_clusters_f=0.7',
'output_path': 'output_ISIC_4_small_clusters_f=0.7'
}
data['ISIC_4_small_clusters_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_small_clusters_f=0.8',
'output_path': 'output_ISIC_4_small_clusters_f=0.8'
}
data['ISIC_4_small_clusters_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_small_clusters_f=0.9',
'output_path': 'output_ISIC_4_small_clusters_f=0.9'
}
data['ISIC_4_small_clusters_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_4',
'dataset_path': 'dataset_ISIC_4_small_clusters_f=1.0',
'output_path': 'output_ISIC_4_small_clusters_f=1.0'
}

data['ISIC_5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5',
  'output_path': 'output_ISIC_5'
  }
data['ISIC_5_image_rot_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_image_rot_f=0.1',
  'output_path': 'output_ISIC_5_image_rot_f=0.1'
  }
data['ISIC_5_image_rot_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_image_rot_f=0.2',
  'output_path': 'output_ISIC_5_image_rot_f=0.2'
  }
data['ISIC_5_image_rot_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_image_rot_f=0.3',
  'output_path': 'output_ISIC_5_image_rot_f=0.3'
  }
data['ISIC_5_image_rot_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_image_rot_f=0.4',
  'output_path': 'output_ISIC_5_image_rot_f=0.4'
  }
data['ISIC_5_image_rot_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_image_rot_f=0.5',
  'output_path': 'output_ISIC_5_image_rot_f=0.5'
  }
data['ISIC_5_image_rot_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_image_rot_f=0.6',
  'output_path': 'output_ISIC_5_image_rot_f=0.6'
  }
data['ISIC_5_image_rot_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_image_rot_f=0.7',
  'output_path': 'output_ISIC_5_image_rot_f=0.7'
  }
data['ISIC_5_image_rot_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_image_rot_f=0.8',
  'output_path': 'output_ISIC_5_image_rot_f=0.8'
  }
data['ISIC_5_image_rot_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_image_rot_f=0.9',
  'output_path': 'output_ISIC_5_image_rot_f=0.9'
  }
data['ISIC_5_image_rot_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_image_rot_f=1.0',
  'output_path': 'output_ISIC_5_image_rot_f=1.0'
  }

data['ISIC_5_image_translation_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_image_translation_f=0.1',
  'output_path': 'output_ISIC_5_image_translation_f=0.1'
  }
data['ISIC_5_image_translation_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_image_translation_f=0.2',
  'output_path': 'output_ISIC_5_image_translation_f=0.2'
  }
data['ISIC_5_image_translation_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_image_translation_f=0.3',
  'output_path': 'output_ISIC_5_image_translation_f=0.3'
  }
data['ISIC_5_image_translation_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_image_translation_f=0.4',
  'output_path': 'output_ISIC_5_image_translation_f=0.4'
  }
data['ISIC_5_image_translation_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_image_translation_f=0.5',
  'output_path': 'output_ISIC_5_image_translation_f=0.5'
  }
data['ISIC_5_image_translation_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_image_translation_f=0.6',
  'output_path': 'output_ISIC_5_image_translation_f=0.6'
  }
data['ISIC_5_image_translation_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_image_translation_f=0.7',
  'output_path': 'output_ISIC_5_image_translation_f=0.7'
  }
data['ISIC_5_image_translation_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_image_translation_f=0.8',
  'output_path': 'output_ISIC_5_image_translation_f=0.8'
  }
data['ISIC_5_image_translation_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_image_translation_f=0.9',
  'output_path': 'output_ISIC_5_image_translation_f=0.9'
  }
data['ISIC_5_image_translation_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_image_translation_f=1.0',
  'output_path': 'output_ISIC_5_image_translation_f=1.0'
  }

data['ISIC_5_image_zoom_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_image_zoom_f=0.1',
  'output_path': 'output_ISIC_5_image_zoom_f=0.1'
  }
data['ISIC_5_image_zoom_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_image_zoom_f=0.2',
  'output_path': 'output_ISIC_5_image_zoom_f=0.2'
  }
data['ISIC_5_image_zoom_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_image_zoom_f=0.3',
  'output_path': 'output_ISIC_5_image_zoom_f=0.3'
  }
data['ISIC_5_image_zoom_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_image_zoom_f=0.4',
  'output_path': 'output_ISIC_5_image_zoom_f=0.4'
  }
data['ISIC_5_image_zoom_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_image_zoom_f=0.5',
  'output_path': 'output_ISIC_5_image_zoom_f=0.5'
  }
data['ISIC_5_image_zoom_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_image_zoom_f=0.6',
  'output_path': 'output_ISIC_5_image_zoom_f=0.6'
  }
data['ISIC_5_image_zoom_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_image_zoom_f=0.7',
  'output_path': 'output_ISIC_5_image_zoom_f=0.7'
  }
data['ISIC_5_image_zoom_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_image_zoom_f=0.8',
  'output_path': 'output_ISIC_5_image_zoom_f=0.8'
  }
data['ISIC_5_image_zoom_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_image_zoom_f=0.9',
  'output_path': 'output_ISIC_5_image_zoom_f=0.9'
  }
data['ISIC_5_image_zoom_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_image_zoom_f=1.0',
  'output_path': 'output_ISIC_5_image_zoom_f=1.0'
  }

data['ISIC_5_add_noise_gaussian_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_gaussian_f=0.1',
  'output_path': 'output_ISIC_5_add_noise_gaussian_f=0.1'
  }
data['ISIC_5_add_noise_gaussian_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_gaussian_f=0.2',
  'output_path': 'output_ISIC_5_add_noise_gaussian_f=0.2'
  }
data['ISIC_5_add_noise_gaussian_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_gaussian_f=0.3',
  'output_path': 'output_ISIC_5_add_noise_gaussian_f=0.3'
  }
data['ISIC_5_add_noise_gaussian_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_gaussian_f=0.4',
  'output_path': 'output_ISIC_5_add_noise_gaussian_f=0.4'
  }
data['ISIC_5_add_noise_gaussian_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_gaussian_f=0.5',
  'output_path': 'output_ISIC_5_add_noise_gaussian_f=0.5'
  }
data['ISIC_5_add_noise_gaussian_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_gaussian_f=0.6',
  'output_path': 'output_ISIC_5_add_noise_gaussian_f=0.6'
  }
data['ISIC_5_add_noise_gaussian_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_gaussian_f=0.7',
  'output_path': 'output_ISIC_5_add_noise_gaussian_f=0.7'
  }
data['ISIC_5_add_noise_gaussian_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_gaussian_f=0.8',
  'output_path': 'output_ISIC_5_add_noise_gaussian_f=0.8'
  }
data['ISIC_5_add_noise_gaussian_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_gaussian_f=0.9',
  'output_path': 'output_ISIC_5_add_noise_gaussian_f=0.9'
  }
data['ISIC_5_add_noise_gaussian_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_gaussian_f=1.0',
  'output_path': 'output_ISIC_5_add_noise_gaussian_f=1.0'
  }

data['ISIC_5_add_noise_poisson_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_poisson_f=0.1',
  'output_path': 'output_ISIC_5_add_noise_poisson_f=0.1'
  }
data['ISIC_5_add_noise_poisson_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_poisson_f=0.2',
  'output_path': 'output_ISIC_5_add_noise_poisson_f=0.2'
  }
data['ISIC_5_add_noise_poisson_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_poisson_f=0.3',
  'output_path': 'output_ISIC_5_add_noise_poisson_f=0.3'
  }
data['ISIC_5_add_noise_poisson_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_poisson_f=0.4',
  'output_path': 'output_ISIC_5_add_noise_poisson_f=0.4'
  }
data['ISIC_5_add_noise_poisson_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_poisson_f=0.5',
  'output_path': 'output_ISIC_5_add_noise_poisson_f=0.5'
  }
data['ISIC_5_add_noise_poisson_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_poisson_f=0.6',
  'output_path': 'output_ISIC_5_add_noise_poisson_f=0.6'
  }
data['ISIC_5_add_noise_poisson_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_poisson_f=0.7',
  'output_path': 'output_ISIC_5_add_noise_poisson_f=0.7'
  }
data['ISIC_5_add_noise_poisson_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_poisson_f=0.8',
  'output_path': 'output_ISIC_5_add_noise_poisson_f=0.8'
  }
data['ISIC_5_add_noise_poisson_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_poisson_f=0.9',
  'output_path': 'output_ISIC_5_add_noise_poisson_f=0.9'
  }
data['ISIC_5_add_noise_poisson_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_poisson_f=1.0',
  'output_path': 'output_ISIC_5_add_noise_poisson_f=1.0'
  }

data['ISIC_5_add_noise_salt_and_pepper_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_salt_and_pepper_f=0.1',
  'output_path': 'output_ISIC_5_add_noise_salt_and_pepper_f=0.1'
  }
data['ISIC_5_add_noise_salt_and_pepper_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_salt_and_pepper_f=0.2',
  'output_path': 'output_ISIC_5_add_noise_salt_and_pepper_f=0.2'
  }
data['ISIC_5_add_noise_salt_and_pepper_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_salt_and_pepper_f=0.3',
  'output_path': 'output_ISIC_5_add_noise_salt_and_pepper_f=0.3'
  }
data['ISIC_5_add_noise_salt_and_pepper_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_salt_and_pepper_f=0.4',
  'output_path': 'output_ISIC_5_add_noise_salt_and_pepper_f=0.4'
  }
data['ISIC_5_add_noise_salt_and_pepper_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_salt_and_pepper_f=0.5',
  'output_path': 'output_ISIC_5_add_noise_salt_and_pepper_f=0.5'
  }
data['ISIC_5_add_noise_salt_and_pepper_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_salt_and_pepper_f=0.6',
  'output_path': 'output_ISIC_5_add_noise_salt_and_pepper_f=0.6'
  }
data['ISIC_5_add_noise_salt_and_pepper_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_salt_and_pepper_f=0.7',
  'output_path': 'output_ISIC_5_add_noise_salt_and_pepper_f=0.7'
  }
data['ISIC_5_add_noise_salt_and_pepper_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_salt_and_pepper_f=0.8',
  'output_path': 'output_ISIC_5_add_noise_salt_and_pepper_f=0.8'
  }
data['ISIC_5_add_noise_salt_and_pepper_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_salt_and_pepper_f=0.9',
  'output_path': 'output_ISIC_5_add_noise_salt_and_pepper_f=0.9'
  }
data['ISIC_5_add_noise_salt_and_pepper_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_salt_and_pepper_f=1.0',
  'output_path': 'output_ISIC_5_add_noise_salt_and_pepper_f=1.0'
  }

data['ISIC_5_add_noise_speckle_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_speckle_f=0.1',
  'output_path': 'output_ISIC_5_add_noise_speckle_f=0.1'
  }
data['ISIC_5_add_noise_speckle_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_speckle_f=0.2',
  'output_path': 'output_ISIC_5_add_noise_speckle_f=0.2'
  }
data['ISIC_5_add_noise_speckle_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_speckle_f=0.3',
  'output_path': 'output_ISIC_5_add_noise_speckle_f=0.3'
  }
data['ISIC_5_add_noise_speckle_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_speckle_f=0.4',
  'output_path': 'output_ISIC_5_add_noise_speckle_f=0.4'
  }
data['ISIC_5_add_noise_speckle_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_speckle_f=0.5',
  'output_path': 'output_ISIC_5_add_noise_speckle_f=0.5'
  }
data['ISIC_5_add_noise_speckle_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_speckle_f=0.6',
  'output_path': 'output_ISIC_5_add_noise_speckle_f=0.6'
  }
data['ISIC_5_add_noise_speckle_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_speckle_f=0.7',
  'output_path': 'output_ISIC_5_add_noise_speckle_f=0.7'
  }
data['ISIC_5_add_noise_speckle_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_speckle_f=0.8',
  'output_path': 'output_ISIC_5_add_noise_speckle_f=0.8'
  }
data['ISIC_5_add_noise_speckle_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_speckle_f=0.9',
  'output_path': 'output_ISIC_5_add_noise_speckle_f=0.9'
  }
data['ISIC_5_add_noise_speckle_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_add_noise_speckle_f=1.0',
  'output_path': 'output_ISIC_5_add_noise_speckle_f=1.0'
  }

data['ISIC_5_imbalance_classes_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_imbalance_classes_f=0.1',
'output_path': 'output_ISIC_5_imbalance_classes_f=0.1'
}
data['ISIC_5_imbalance_classes_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_imbalance_classes_f=0.2',
'output_path': 'output_ISIC_5_imbalance_classes_f=0.2'
}
data['ISIC_5_imbalance_classes_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_imbalance_classes_f=0.3',
'output_path': 'output_ISIC_5_imbalance_classes_f=0.3'
}
data['ISIC_5_imbalance_classes_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_imbalance_classes_f=0.4',
'output_path': 'output_ISIC_5_imbalance_classes_f=0.4'
}
data['ISIC_5_imbalance_classes_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_imbalance_classes_f=0.5',
'output_path': 'output_ISIC_5_imbalance_classes_f=0.5'
}
data['ISIC_5_imbalance_classes_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_imbalance_classes_f=0.6',
'output_path': 'output_ISIC_5_imbalance_classes_f=0.6'
}
data['ISIC_5_imbalance_classes_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_imbalance_classes_f=0.7',
'output_path': 'output_ISIC_5_imbalance_classes_f=0.7'
}
data['ISIC_5_imbalance_classes_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_imbalance_classes_f=0.8',
'output_path': 'output_ISIC_5_imbalance_classes_f=0.8'
}
data['ISIC_5_imbalance_classes_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_imbalance_classes_f=0.9',
'output_path': 'output_ISIC_5_imbalance_classes_f=0.9'
}
data['ISIC_5_imbalance_classes_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_imbalance_classes_f=1.0',
'output_path': 'output_ISIC_5_imbalance_classes_f=1.0'
}

data['ISIC_5_grayscale_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_grayscale_f=0.1',
'output_path': 'output_ISIC_5_grayscale_f=0.1'
}
data['ISIC_5_grayscale_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_grayscale_f=0.2',
'output_path': 'output_ISIC_5_grayscale_f=0.2'
}
data['ISIC_5_grayscale_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_grayscale_f=0.3',
'output_path': 'output_ISIC_5_grayscale_f=0.3'
}
data['ISIC_5_grayscale_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_grayscale_f=0.4',
'output_path': 'output_ISIC_5_grayscale_f=0.4'
}
data['ISIC_5_grayscale_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_grayscale_f=0.5',
'output_path': 'output_ISIC_5_grayscale_f=0.5'
}
data['ISIC_5_grayscale_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_grayscale_f=0.6',
'output_path': 'output_ISIC_5_grayscale_f=0.6'
}
data['ISIC_5_grayscale_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_grayscale_f=0.7',
'output_path': 'output_ISIC_5_grayscale_f=0.7'
}
data['ISIC_5_grayscale_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_grayscale_f=0.8',
'output_path': 'output_ISIC_5_grayscale_f=0.8'
}
data['ISIC_5_grayscale_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_grayscale_f=0.9',
'output_path': 'output_ISIC_5_grayscale_f=0.9'
}
data['ISIC_5_grayscale_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_grayscale_f=1.0',
'output_path': 'output_ISIC_5_grayscale_f=1.0'
}

data['ISIC_5_hsv_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_hsv_f=0.1',
'output_path': 'output_ISIC_5_hsv_f=0.1'
}
data['ISIC_5_hsv_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_hsv_f=0.2',
'output_path': 'output_ISIC_5_hsv_f=0.2'
}
data['ISIC_5_hsv_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_hsv_f=0.3',
'output_path': 'output_ISIC_5_hsv_f=0.3'
}
data['ISIC_5_hsv_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_hsv_f=0.4',
'output_path': 'output_ISIC_5_hsv_f=0.4'
}
data['ISIC_5_hsv_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_hsv_f=0.5',
'output_path': 'output_ISIC_5_hsv_f=0.5'
}
data['ISIC_5_hsv_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_hsv_f=0.6',
'output_path': 'output_ISIC_5_hsv_f=0.6'
}
data['ISIC_5_hsv_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_hsv_f=0.7',
'output_path': 'output_ISIC_5_hsv_f=0.7'
}
data['ISIC_5_hsv_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_hsv_f=0.8',
'output_path': 'output_ISIC_5_hsv_f=0.8'
}
data['ISIC_5_hsv_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_hsv_f=0.9',
'output_path': 'output_ISIC_5_hsv_f=0.9'
}
data['ISIC_5_hsv_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_hsv_f=1.0',
'output_path': 'output_ISIC_5_hsv_f=1.0'
}
data['ISIC_5_blur_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_blur_f=0.1',
'output_path': 'output_ISIC_5_blur_f=0.1'
}
data['ISIC_5_blur_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_blur_f=0.2',
'output_path': 'output_ISIC_5_blur_f=0.2'
}
data['ISIC_5_blur_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_blur_f=0.3',
'output_path': 'output_ISIC_5_blur_f=0.3'
}
data['ISIC_5_blur_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_blur_f=0.4',
'output_path': 'output_ISIC_5_blur_f=0.4'
}
data['ISIC_5_blur_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_blur_f=0.5',
'output_path': 'output_ISIC_5_blur_f=0.5'
}
data['ISIC_5_blur_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_blur_f=0.6',
'output_path': 'output_ISIC_5_blur_f=0.6'
}
data['ISIC_5_blur_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_blur_f=0.7',
'output_path': 'output_ISIC_5_blur_f=0.7'
}
data['ISIC_5_blur_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_blur_f=0.8',
'output_path': 'output_ISIC_5_blur_f=0.8'
}
data['ISIC_5_blur_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_blur_f=0.9',
'output_path': 'output_ISIC_5_blur_f=0.9'
}
data['ISIC_5_blur_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_blur_f=1.0',
'output_path': 'output_ISIC_5_blur_f=1.0'
}

data['ISIC_5_small_random_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_small_random_f=0.1',
'output_path': 'output_ISIC_5_small_random_f=0.1'
}
data['ISIC_5_small_random_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_small_random_f=0.2',
'output_path': 'output_ISIC_5_small_random_f=0.2'
}
data['ISIC_5_small_random_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_small_random_f=0.3',
'output_path': 'output_ISIC_5_small_random_f=0.3'
}
data['ISIC_5_small_random_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_small_random_f=0.4',
'output_path': 'output_ISIC_5_small_random_f=0.4'
}
data['ISIC_5_small_random_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_small_random_f=0.5',
'output_path': 'output_ISIC_5_small_random_f=0.5'
}
data['ISIC_5_small_random_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_small_random_f=0.6',
'output_path': 'output_ISIC_5_small_random_f=0.6'
}
data['ISIC_5_small_random_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_small_random_f=0.7',
'output_path': 'output_ISIC_5_small_random_f=0.7'
}
data['ISIC_5_small_random_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_small_random_f=0.8',
'output_path': 'output_ISIC_5_small_random_f=0.8'
}
data['ISIC_5_small_random_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_small_random_f=0.9',
'output_path': 'output_ISIC_5_small_random_f=0.9'
}
data['ISIC_5_small_random_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_small_random_f=1.0',
'output_path': 'output_ISIC_5_small_random_f=1.0'
}

data['ISIC_5_small_easy_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_small_easy_f=0.1',
  'output_path': 'output_ISIC_5_small_easy_f=0.1'
  }
data['ISIC_5_small_easy_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_small_easy_f=0.2',
  'output_path': 'output_ISIC_5_small_easy_f=0.2'
  }
data['ISIC_5_small_easy_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_small_easy_f=0.3',
  'output_path': 'output_ISIC_5_small_easy_f=0.3'
  }
data['ISIC_5_small_easy_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_small_easy_f=0.4',
  'output_path': 'output_ISIC_5_small_easy_f=0.4'
  }
data['ISIC_5_small_easy_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_small_easy_f=0.5',
  'output_path': 'output_ISIC_5_small_easy_f=0.5'
  }
data['ISIC_5_small_easy_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_small_easy_f=0.6',
  'output_path': 'output_ISIC_5_small_easy_f=0.6'
  }
data['ISIC_5_small_easy_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_small_easy_f=0.7',
  'output_path': 'output_ISIC_5_small_easy_f=0.7'
  }
data['ISIC_5_small_easy_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_small_easy_f=0.8',
  'output_path': 'output_ISIC_5_small_easy_f=0.8'
  }
data['ISIC_5_small_easy_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_small_easy_f=0.9',
  'output_path': 'output_ISIC_5_small_easy_f=0.9'
  }
data['ISIC_5_small_easy_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_small_easy_f=1.0',
  'output_path': 'output_ISIC_5_small_easy_f=1.0'
  }

data['ISIC_5_small_hard_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_small_hard_f=0.1',
  'output_path': 'output_ISIC_5_small_hard_f=0.1'
  }
data['ISIC_5_small_hard_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_small_hard_f=0.2',
  'output_path': 'output_ISIC_5_small_hard_f=0.2'
  }
data['ISIC_5_small_hard_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_small_hard_f=0.3',
  'output_path': 'output_ISIC_5_small_hard_f=0.3'
  }
data['ISIC_5_small_hard_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_small_hard_f=0.4',
  'output_path': 'output_ISIC_5_small_hard_f=0.4'
  }
data['ISIC_5_small_hard_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_small_hard_f=0.5',
  'output_path': 'output_ISIC_5_small_hard_f=0.5'
  }
data['ISIC_5_small_hard_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_small_hard_f=0.6',
  'output_path': 'output_ISIC_5_small_hard_f=0.6'
  }
data['ISIC_5_small_hard_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_small_hard_f=0.7',
  'output_path': 'output_ISIC_5_small_hard_f=0.7'
  }
data['ISIC_5_small_hard_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_small_hard_f=0.8',
  'output_path': 'output_ISIC_5_small_hard_f=0.8'
  }
data['ISIC_5_small_hard_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_small_hard_f=0.9',
  'output_path': 'output_ISIC_5_small_hard_f=0.9'
  }
data['ISIC_5_small_hard_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_5',
  'dataset_path': 'dataset_ISIC_5_small_hard_f=1.0',
  'output_path': 'output_ISIC_5_small_hard_f=1.0'
  }

data['ISIC_5_small_clusters_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_small_clusters_f=0.1',
'output_path': 'output_ISIC_5_small_clusters_f=0.1'
}
data['ISIC_5_small_clusters_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_small_clusters_f=0.2',
'output_path': 'output_ISIC_5_small_clusters_f=0.2'
}
data['ISIC_5_small_clusters_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_small_clusters_f=0.3',
'output_path': 'output_ISIC_5_small_clusters_f=0.3'
}
data['ISIC_5_small_clusters_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_small_clusters_f=0.4',
'output_path': 'output_ISIC_5_small_clusters_f=0.4'
}
data['ISIC_5_small_clusters_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_small_clusters_f=0.5',
'output_path': 'output_ISIC_5_small_clusters_f=0.5'
}
data['ISIC_5_small_clusters_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_small_clusters_f=0.6',
'output_path': 'output_ISIC_5_small_clusters_f=0.6'
}
data['ISIC_5_small_clusters_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_small_clusters_f=0.7',
'output_path': 'output_ISIC_5_small_clusters_f=0.7'
}
data['ISIC_5_small_clusters_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_small_clusters_f=0.8',
'output_path': 'output_ISIC_5_small_clusters_f=0.8'
}
data['ISIC_5_small_clusters_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_small_clusters_f=0.9',
'output_path': 'output_ISIC_5_small_clusters_f=0.9'
}
data['ISIC_5_small_clusters_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_5',
'dataset_path': 'dataset_ISIC_5_small_clusters_f=1.0',
'output_path': 'output_ISIC_5_small_clusters_f=1.0'
}

data['ISIC_6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6',
  'output_path': 'output_ISIC_6'
  }
data['ISIC_6_image_rot_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_image_rot_f=0.1',
  'output_path': 'output_ISIC_6_image_rot_f=0.1'
  }
data['ISIC_6_image_rot_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_image_rot_f=0.2',
  'output_path': 'output_ISIC_6_image_rot_f=0.2'
  }
data['ISIC_6_image_rot_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_image_rot_f=0.3',
  'output_path': 'output_ISIC_6_image_rot_f=0.3'
  }
data['ISIC_6_image_rot_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_image_rot_f=0.4',
  'output_path': 'output_ISIC_6_image_rot_f=0.4'
  }
data['ISIC_6_image_rot_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_image_rot_f=0.5',
  'output_path': 'output_ISIC_6_image_rot_f=0.5'
  }
data['ISIC_6_image_rot_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_image_rot_f=0.6',
  'output_path': 'output_ISIC_6_image_rot_f=0.6'
  }
data['ISIC_6_image_rot_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_image_rot_f=0.7',
  'output_path': 'output_ISIC_6_image_rot_f=0.7'
  }
data['ISIC_6_image_rot_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_image_rot_f=0.8',
  'output_path': 'output_ISIC_6_image_rot_f=0.8'
  }
data['ISIC_6_image_rot_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_image_rot_f=0.9',
  'output_path': 'output_ISIC_6_image_rot_f=0.9'
  }
data['ISIC_6_image_rot_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_image_rot_f=1.0',
  'output_path': 'output_ISIC_6_image_rot_f=1.0'
  }

data['ISIC_6_image_translation_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_image_translation_f=0.1',
  'output_path': 'output_ISIC_6_image_translation_f=0.1'
  }
data['ISIC_6_image_translation_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_image_translation_f=0.2',
  'output_path': 'output_ISIC_6_image_translation_f=0.2'
  }
data['ISIC_6_image_translation_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_image_translation_f=0.3',
  'output_path': 'output_ISIC_6_image_translation_f=0.3'
  }
data['ISIC_6_image_translation_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_image_translation_f=0.4',
  'output_path': 'output_ISIC_6_image_translation_f=0.4'
  }
data['ISIC_6_image_translation_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_image_translation_f=0.5',
  'output_path': 'output_ISIC_6_image_translation_f=0.5'
  }
data['ISIC_6_image_translation_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_image_translation_f=0.6',
  'output_path': 'output_ISIC_6_image_translation_f=0.6'
  }
data['ISIC_6_image_translation_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_image_translation_f=0.7',
  'output_path': 'output_ISIC_6_image_translation_f=0.7'
  }
data['ISIC_6_image_translation_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_image_translation_f=0.8',
  'output_path': 'output_ISIC_6_image_translation_f=0.8'
  }
data['ISIC_6_image_translation_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_image_translation_f=0.9',
  'output_path': 'output_ISIC_6_image_translation_f=0.9'
  }
data['ISIC_6_image_translation_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_image_translation_f=1.0',
  'output_path': 'output_ISIC_6_image_translation_f=1.0'
  }

data['ISIC_6_image_zoom_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_image_zoom_f=0.1',
  'output_path': 'output_ISIC_6_image_zoom_f=0.1'
  }
data['ISIC_6_image_zoom_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_image_zoom_f=0.2',
  'output_path': 'output_ISIC_6_image_zoom_f=0.2'
  }
data['ISIC_6_image_zoom_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_image_zoom_f=0.3',
  'output_path': 'output_ISIC_6_image_zoom_f=0.3'
  }
data['ISIC_6_image_zoom_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_image_zoom_f=0.4',
  'output_path': 'output_ISIC_6_image_zoom_f=0.4'
  }
data['ISIC_6_image_zoom_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_image_zoom_f=0.5',
  'output_path': 'output_ISIC_6_image_zoom_f=0.5'
  }
data['ISIC_6_image_zoom_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_image_zoom_f=0.6',
  'output_path': 'output_ISIC_6_image_zoom_f=0.6'
  }
data['ISIC_6_image_zoom_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_image_zoom_f=0.7',
  'output_path': 'output_ISIC_6_image_zoom_f=0.7'
  }
data['ISIC_6_image_zoom_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_image_zoom_f=0.8',
  'output_path': 'output_ISIC_6_image_zoom_f=0.8'
  }
data['ISIC_6_image_zoom_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_image_zoom_f=0.9',
  'output_path': 'output_ISIC_6_image_zoom_f=0.9'
  }
data['ISIC_6_image_zoom_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_image_zoom_f=1.0',
  'output_path': 'output_ISIC_6_image_zoom_f=1.0'
  }

data['ISIC_6_add_noise_gaussian_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_gaussian_f=0.1',
  'output_path': 'output_ISIC_6_add_noise_gaussian_f=0.1'
  }
data['ISIC_6_add_noise_gaussian_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_gaussian_f=0.2',
  'output_path': 'output_ISIC_6_add_noise_gaussian_f=0.2'
  }
data['ISIC_6_add_noise_gaussian_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_gaussian_f=0.3',
  'output_path': 'output_ISIC_6_add_noise_gaussian_f=0.3'
  }
data['ISIC_6_add_noise_gaussian_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_gaussian_f=0.4',
  'output_path': 'output_ISIC_6_add_noise_gaussian_f=0.4'
  }
data['ISIC_6_add_noise_gaussian_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_gaussian_f=0.5',
  'output_path': 'output_ISIC_6_add_noise_gaussian_f=0.5'
  }
data['ISIC_6_add_noise_gaussian_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_gaussian_f=0.6',
  'output_path': 'output_ISIC_6_add_noise_gaussian_f=0.6'
  }
data['ISIC_6_add_noise_gaussian_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_gaussian_f=0.7',
  'output_path': 'output_ISIC_6_add_noise_gaussian_f=0.7'
  }
data['ISIC_6_add_noise_gaussian_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_gaussian_f=0.8',
  'output_path': 'output_ISIC_6_add_noise_gaussian_f=0.8'
  }
data['ISIC_6_add_noise_gaussian_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_gaussian_f=0.9',
  'output_path': 'output_ISIC_6_add_noise_gaussian_f=0.9'
  }
data['ISIC_6_add_noise_gaussian_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_gaussian_f=1.0',
  'output_path': 'output_ISIC_6_add_noise_gaussian_f=1.0'
  }

data['ISIC_6_add_noise_poisson_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_poisson_f=0.1',
  'output_path': 'output_ISIC_6_add_noise_poisson_f=0.1'
  }
data['ISIC_6_add_noise_poisson_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_poisson_f=0.2',
  'output_path': 'output_ISIC_6_add_noise_poisson_f=0.2'
  }
data['ISIC_6_add_noise_poisson_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_poisson_f=0.3',
  'output_path': 'output_ISIC_6_add_noise_poisson_f=0.3'
  }
data['ISIC_6_add_noise_poisson_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_poisson_f=0.4',
  'output_path': 'output_ISIC_6_add_noise_poisson_f=0.4'
  }
data['ISIC_6_add_noise_poisson_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_poisson_f=0.5',
  'output_path': 'output_ISIC_6_add_noise_poisson_f=0.5'
  }
data['ISIC_6_add_noise_poisson_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_poisson_f=0.6',
  'output_path': 'output_ISIC_6_add_noise_poisson_f=0.6'
  }
data['ISIC_6_add_noise_poisson_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_poisson_f=0.7',
  'output_path': 'output_ISIC_6_add_noise_poisson_f=0.7'
  }
data['ISIC_6_add_noise_poisson_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_poisson_f=0.8',
  'output_path': 'output_ISIC_6_add_noise_poisson_f=0.8'
  }
data['ISIC_6_add_noise_poisson_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_poisson_f=0.9',
  'output_path': 'output_ISIC_6_add_noise_poisson_f=0.9'
  }
data['ISIC_6_add_noise_poisson_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_poisson_f=1.0',
  'output_path': 'output_ISIC_6_add_noise_poisson_f=1.0'
  }

data['ISIC_6_add_noise_salt_and_pepper_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_salt_and_pepper_f=0.1',
  'output_path': 'output_ISIC_6_add_noise_salt_and_pepper_f=0.1'
  }
data['ISIC_6_add_noise_salt_and_pepper_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_salt_and_pepper_f=0.2',
  'output_path': 'output_ISIC_6_add_noise_salt_and_pepper_f=0.2'
  }
data['ISIC_6_add_noise_salt_and_pepper_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_salt_and_pepper_f=0.3',
  'output_path': 'output_ISIC_6_add_noise_salt_and_pepper_f=0.3'
  }
data['ISIC_6_add_noise_salt_and_pepper_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_salt_and_pepper_f=0.4',
  'output_path': 'output_ISIC_6_add_noise_salt_and_pepper_f=0.4'
  }
data['ISIC_6_add_noise_salt_and_pepper_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_salt_and_pepper_f=0.5',
  'output_path': 'output_ISIC_6_add_noise_salt_and_pepper_f=0.5'
  }
data['ISIC_6_add_noise_salt_and_pepper_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_salt_and_pepper_f=0.6',
  'output_path': 'output_ISIC_6_add_noise_salt_and_pepper_f=0.6'
  }
data['ISIC_6_add_noise_salt_and_pepper_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_salt_and_pepper_f=0.7',
  'output_path': 'output_ISIC_6_add_noise_salt_and_pepper_f=0.7'
  }
data['ISIC_6_add_noise_salt_and_pepper_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_salt_and_pepper_f=0.8',
  'output_path': 'output_ISIC_6_add_noise_salt_and_pepper_f=0.8'
  }
data['ISIC_6_add_noise_salt_and_pepper_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_salt_and_pepper_f=0.9',
  'output_path': 'output_ISIC_6_add_noise_salt_and_pepper_f=0.9'
  }
data['ISIC_6_add_noise_salt_and_pepper_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_salt_and_pepper_f=1.0',
  'output_path': 'output_ISIC_6_add_noise_salt_and_pepper_f=1.0'
  }

data['ISIC_6_add_noise_speckle_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_speckle_f=0.1',
  'output_path': 'output_ISIC_6_add_noise_speckle_f=0.1'
  }
data['ISIC_6_add_noise_speckle_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_speckle_f=0.2',
  'output_path': 'output_ISIC_6_add_noise_speckle_f=0.2'
  }
data['ISIC_6_add_noise_speckle_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_speckle_f=0.3',
  'output_path': 'output_ISIC_6_add_noise_speckle_f=0.3'
  }
data['ISIC_6_add_noise_speckle_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_speckle_f=0.4',
  'output_path': 'output_ISIC_6_add_noise_speckle_f=0.4'
  }
data['ISIC_6_add_noise_speckle_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_speckle_f=0.5',
  'output_path': 'output_ISIC_6_add_noise_speckle_f=0.5'
  }
data['ISIC_6_add_noise_speckle_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_speckle_f=0.6',
  'output_path': 'output_ISIC_6_add_noise_speckle_f=0.6'
  }
data['ISIC_6_add_noise_speckle_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_speckle_f=0.7',
  'output_path': 'output_ISIC_6_add_noise_speckle_f=0.7'
  }
data['ISIC_6_add_noise_speckle_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_speckle_f=0.8',
  'output_path': 'output_ISIC_6_add_noise_speckle_f=0.8'
  }
data['ISIC_6_add_noise_speckle_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_speckle_f=0.9',
  'output_path': 'output_ISIC_6_add_noise_speckle_f=0.9'
  }
data['ISIC_6_add_noise_speckle_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_add_noise_speckle_f=1.0',
  'output_path': 'output_ISIC_6_add_noise_speckle_f=1.0'
  }

data['ISIC_6_imbalance_classes_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_imbalance_classes_f=0.1',
'output_path': 'output_ISIC_6_imbalance_classes_f=0.1'
}
data['ISIC_6_imbalance_classes_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_imbalance_classes_f=0.2',
'output_path': 'output_ISIC_6_imbalance_classes_f=0.2'
}
data['ISIC_6_imbalance_classes_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_imbalance_classes_f=0.3',
'output_path': 'output_ISIC_6_imbalance_classes_f=0.3'
}
data['ISIC_6_imbalance_classes_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_imbalance_classes_f=0.4',
'output_path': 'output_ISIC_6_imbalance_classes_f=0.4'
}
data['ISIC_6_imbalance_classes_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_imbalance_classes_f=0.5',
'output_path': 'output_ISIC_6_imbalance_classes_f=0.5'
}
data['ISIC_6_imbalance_classes_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_imbalance_classes_f=0.6',
'output_path': 'output_ISIC_6_imbalance_classes_f=0.6'
}
data['ISIC_6_imbalance_classes_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_imbalance_classes_f=0.7',
'output_path': 'output_ISIC_6_imbalance_classes_f=0.7'
}
data['ISIC_6_imbalance_classes_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_imbalance_classes_f=0.8',
'output_path': 'output_ISIC_6_imbalance_classes_f=0.8'
}
data['ISIC_6_imbalance_classes_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_imbalance_classes_f=0.9',
'output_path': 'output_ISIC_6_imbalance_classes_f=0.9'
}
data['ISIC_6_imbalance_classes_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_imbalance_classes_f=1.0',
'output_path': 'output_ISIC_6_imbalance_classes_f=1.0'
}

data['ISIC_6_grayscale_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_grayscale_f=0.1',
'output_path': 'output_ISIC_6_grayscale_f=0.1'
}
data['ISIC_6_grayscale_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_grayscale_f=0.2',
'output_path': 'output_ISIC_6_grayscale_f=0.2'
}
data['ISIC_6_grayscale_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_grayscale_f=0.3',
'output_path': 'output_ISIC_6_grayscale_f=0.3'
}
data['ISIC_6_grayscale_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_grayscale_f=0.4',
'output_path': 'output_ISIC_6_grayscale_f=0.4'
}
data['ISIC_6_grayscale_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_grayscale_f=0.5',
'output_path': 'output_ISIC_6_grayscale_f=0.5'
}
data['ISIC_6_grayscale_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_grayscale_f=0.6',
'output_path': 'output_ISIC_6_grayscale_f=0.6'
}
data['ISIC_6_grayscale_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_grayscale_f=0.7',
'output_path': 'output_ISIC_6_grayscale_f=0.7'
}
data['ISIC_6_grayscale_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_grayscale_f=0.8',
'output_path': 'output_ISIC_6_grayscale_f=0.8'
}
data['ISIC_6_grayscale_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_grayscale_f=0.9',
'output_path': 'output_ISIC_6_grayscale_f=0.9'
}
data['ISIC_6_grayscale_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_grayscale_f=1.0',
'output_path': 'output_ISIC_6_grayscale_f=1.0'
}

data['ISIC_6_hsv_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_hsv_f=0.1',
'output_path': 'output_ISIC_6_hsv_f=0.1'
}
data['ISIC_6_hsv_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_hsv_f=0.2',
'output_path': 'output_ISIC_6_hsv_f=0.2'
}
data['ISIC_6_hsv_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_hsv_f=0.3',
'output_path': 'output_ISIC_6_hsv_f=0.3'
}
data['ISIC_6_hsv_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_hsv_f=0.4',
'output_path': 'output_ISIC_6_hsv_f=0.4'
}
data['ISIC_6_hsv_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_hsv_f=0.5',
'output_path': 'output_ISIC_6_hsv_f=0.5'
}
data['ISIC_6_hsv_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_hsv_f=0.6',
'output_path': 'output_ISIC_6_hsv_f=0.6'
}
data['ISIC_6_hsv_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_hsv_f=0.7',
'output_path': 'output_ISIC_6_hsv_f=0.7'
}
data['ISIC_6_hsv_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_hsv_f=0.8',
'output_path': 'output_ISIC_6_hsv_f=0.8'
}
data['ISIC_6_hsv_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_hsv_f=0.9',
'output_path': 'output_ISIC_6_hsv_f=0.9'
}
data['ISIC_6_hsv_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_hsv_f=1.0',
'output_path': 'output_ISIC_6_hsv_f=1.0'
}
data['ISIC_6_blur_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_blur_f=0.1',
'output_path': 'output_ISIC_6_blur_f=0.1'
}
data['ISIC_6_blur_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_blur_f=0.2',
'output_path': 'output_ISIC_6_blur_f=0.2'
}
data['ISIC_6_blur_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_blur_f=0.3',
'output_path': 'output_ISIC_6_blur_f=0.3'
}
data['ISIC_6_blur_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_blur_f=0.4',
'output_path': 'output_ISIC_6_blur_f=0.4'
}
data['ISIC_6_blur_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_blur_f=0.5',
'output_path': 'output_ISIC_6_blur_f=0.5'
}
data['ISIC_6_blur_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_blur_f=0.6',
'output_path': 'output_ISIC_6_blur_f=0.6'
}
data['ISIC_6_blur_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_blur_f=0.7',
'output_path': 'output_ISIC_6_blur_f=0.7'
}
data['ISIC_6_blur_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_blur_f=0.8',
'output_path': 'output_ISIC_6_blur_f=0.8'
}
data['ISIC_6_blur_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_blur_f=0.9',
'output_path': 'output_ISIC_6_blur_f=0.9'
}
data['ISIC_6_blur_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_blur_f=1.0',
'output_path': 'output_ISIC_6_blur_f=1.0'
}

data['ISIC_6_small_random_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_small_random_f=0.1',
'output_path': 'output_ISIC_6_small_random_f=0.1'
}
data['ISIC_6_small_random_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_small_random_f=0.2',
'output_path': 'output_ISIC_6_small_random_f=0.2'
}
data['ISIC_6_small_random_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_small_random_f=0.3',
'output_path': 'output_ISIC_6_small_random_f=0.3'
}
data['ISIC_6_small_random_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_small_random_f=0.4',
'output_path': 'output_ISIC_6_small_random_f=0.4'
}
data['ISIC_6_small_random_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_small_random_f=0.5',
'output_path': 'output_ISIC_6_small_random_f=0.5'
}
data['ISIC_6_small_random_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_small_random_f=0.6',
'output_path': 'output_ISIC_6_small_random_f=0.6'
}
data['ISIC_6_small_random_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_small_random_f=0.7',
'output_path': 'output_ISIC_6_small_random_f=0.7'
}
data['ISIC_6_small_random_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_small_random_f=0.8',
'output_path': 'output_ISIC_6_small_random_f=0.8'
}
data['ISIC_6_small_random_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_small_random_f=0.9',
'output_path': 'output_ISIC_6_small_random_f=0.9'
}
data['ISIC_6_small_random_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_small_random_f=1.0',
'output_path': 'output_ISIC_6_small_random_f=1.0'
}

data['ISIC_6_small_easy_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_small_easy_f=0.1',
  'output_path': 'output_ISIC_6_small_easy_f=0.1'
  }
data['ISIC_6_small_easy_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_small_easy_f=0.2',
  'output_path': 'output_ISIC_6_small_easy_f=0.2'
  }
data['ISIC_6_small_easy_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_small_easy_f=0.3',
  'output_path': 'output_ISIC_6_small_easy_f=0.3'
  }
data['ISIC_6_small_easy_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_small_easy_f=0.4',
  'output_path': 'output_ISIC_6_small_easy_f=0.4'
  }
data['ISIC_6_small_easy_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_small_easy_f=0.5',
  'output_path': 'output_ISIC_6_small_easy_f=0.5'
  }
data['ISIC_6_small_easy_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_small_easy_f=0.6',
  'output_path': 'output_ISIC_6_small_easy_f=0.6'
  }
data['ISIC_6_small_easy_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_small_easy_f=0.7',
  'output_path': 'output_ISIC_6_small_easy_f=0.7'
  }
data['ISIC_6_small_easy_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_small_easy_f=0.8',
  'output_path': 'output_ISIC_6_small_easy_f=0.8'
  }
data['ISIC_6_small_easy_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_small_easy_f=0.9',
  'output_path': 'output_ISIC_6_small_easy_f=0.9'
  }
data['ISIC_6_small_easy_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_small_easy_f=1.0',
  'output_path': 'output_ISIC_6_small_easy_f=1.0'
  }

data['ISIC_6_small_hard_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_small_hard_f=0.1',
  'output_path': 'output_ISIC_6_small_hard_f=0.1'
  }
data['ISIC_6_small_hard_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_small_hard_f=0.2',
  'output_path': 'output_ISIC_6_small_hard_f=0.2'
  }
data['ISIC_6_small_hard_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_small_hard_f=0.3',
  'output_path': 'output_ISIC_6_small_hard_f=0.3'
  }
data['ISIC_6_small_hard_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_small_hard_f=0.4',
  'output_path': 'output_ISIC_6_small_hard_f=0.4'
  }
data['ISIC_6_small_hard_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_small_hard_f=0.5',
  'output_path': 'output_ISIC_6_small_hard_f=0.5'
  }
data['ISIC_6_small_hard_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_small_hard_f=0.6',
  'output_path': 'output_ISIC_6_small_hard_f=0.6'
  }
data['ISIC_6_small_hard_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_small_hard_f=0.7',
  'output_path': 'output_ISIC_6_small_hard_f=0.7'
  }
data['ISIC_6_small_hard_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_small_hard_f=0.8',
  'output_path': 'output_ISIC_6_small_hard_f=0.8'
  }
data['ISIC_6_small_hard_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_small_hard_f=0.9',
  'output_path': 'output_ISIC_6_small_hard_f=0.9'
  }
data['ISIC_6_small_hard_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'ISIC_6',
  'dataset_path': 'dataset_ISIC_6_small_hard_f=1.0',
  'output_path': 'output_ISIC_6_small_hard_f=1.0'
  }

data['ISIC_6_small_clusters_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_small_clusters_f=0.1',
'output_path': 'output_ISIC_6_small_clusters_f=0.1'
}
data['ISIC_6_small_clusters_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_small_clusters_f=0.2',
'output_path': 'output_ISIC_6_small_clusters_f=0.2'
}
data['ISIC_6_small_clusters_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_small_clusters_f=0.3',
'output_path': 'output_ISIC_6_small_clusters_f=0.3'
}
data['ISIC_6_small_clusters_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_small_clusters_f=0.4',
'output_path': 'output_ISIC_6_small_clusters_f=0.4'
}
data['ISIC_6_small_clusters_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_small_clusters_f=0.5',
'output_path': 'output_ISIC_6_small_clusters_f=0.5'
}
data['ISIC_6_small_clusters_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_small_clusters_f=0.6',
'output_path': 'output_ISIC_6_small_clusters_f=0.6'
}
data['ISIC_6_small_clusters_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_small_clusters_f=0.7',
'output_path': 'output_ISIC_6_small_clusters_f=0.7'
}
data['ISIC_6_small_clusters_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_small_clusters_f=0.8',
'output_path': 'output_ISIC_6_small_clusters_f=0.8'
}
data['ISIC_6_small_clusters_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_small_clusters_f=0.9',
'output_path': 'output_ISIC_6_small_clusters_f=0.9'
}
data['ISIC_6_small_clusters_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'ISIC_6',
'dataset_path': 'dataset_ISIC_6_small_clusters_f=1.0',
'output_path': 'output_ISIC_6_small_clusters_f=1.0'
}

data['CNMC_2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2',
  'output_path': 'output_CNMC_2'
  }
data['CNMC_2_image_rot_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_image_rot_f=0.1',
  'output_path': 'output_CNMC_2_image_rot_f=0.1'
  }
data['CNMC_2_image_rot_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_image_rot_f=0.2',
  'output_path': 'output_CNMC_2_image_rot_f=0.2'
  }
data['CNMC_2_image_rot_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_image_rot_f=0.3',
  'output_path': 'output_CNMC_2_image_rot_f=0.3'
  }
data['CNMC_2_image_rot_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_image_rot_f=0.4',
  'output_path': 'output_CNMC_2_image_rot_f=0.4'
  }
data['CNMC_2_image_rot_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_image_rot_f=0.5',
  'output_path': 'output_CNMC_2_image_rot_f=0.5'
  }
data['CNMC_2_image_rot_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_image_rot_f=0.6',
  'output_path': 'output_CNMC_2_image_rot_f=0.6'
  }
data['CNMC_2_image_rot_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_image_rot_f=0.7',
  'output_path': 'output_CNMC_2_image_rot_f=0.7'
  }
data['CNMC_2_image_rot_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_image_rot_f=0.8',
  'output_path': 'output_CNMC_2_image_rot_f=0.8'
  }
data['CNMC_2_image_rot_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_image_rot_f=0.9',
  'output_path': 'output_CNMC_2_image_rot_f=0.9'
  }
data['CNMC_2_image_rot_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_image_rot_f=1.0',
  'output_path': 'output_CNMC_2_image_rot_f=1.0'
  }

data['CNMC_2_image_translation_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_image_translation_f=0.1',
  'output_path': 'output_CNMC_2_image_translation_f=0.1'
  }
data['CNMC_2_image_translation_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_image_translation_f=0.2',
  'output_path': 'output_CNMC_2_image_translation_f=0.2'
  }
data['CNMC_2_image_translation_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_image_translation_f=0.3',
  'output_path': 'output_CNMC_2_image_translation_f=0.3'
  }
data['CNMC_2_image_translation_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_image_translation_f=0.4',
  'output_path': 'output_CNMC_2_image_translation_f=0.4'
  }
data['CNMC_2_image_translation_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_image_translation_f=0.5',
  'output_path': 'output_CNMC_2_image_translation_f=0.5'
  }
data['CNMC_2_image_translation_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_image_translation_f=0.6',
  'output_path': 'output_CNMC_2_image_translation_f=0.6'
  }
data['CNMC_2_image_translation_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_image_translation_f=0.7',
  'output_path': 'output_CNMC_2_image_translation_f=0.7'
  }
data['CNMC_2_image_translation_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_image_translation_f=0.8',
  'output_path': 'output_CNMC_2_image_translation_f=0.8'
  }
data['CNMC_2_image_translation_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_image_translation_f=0.9',
  'output_path': 'output_CNMC_2_image_translation_f=0.9'
  }
data['CNMC_2_image_translation_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_image_translation_f=1.0',
  'output_path': 'output_CNMC_2_image_translation_f=1.0'
  }

data['CNMC_2_image_zoom_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_image_zoom_f=0.1',
  'output_path': 'output_CNMC_2_image_zoom_f=0.1'
  }
data['CNMC_2_image_zoom_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_image_zoom_f=0.2',
  'output_path': 'output_CNMC_2_image_zoom_f=0.2'
  }
data['CNMC_2_image_zoom_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_image_zoom_f=0.3',
  'output_path': 'output_CNMC_2_image_zoom_f=0.3'
  }
data['CNMC_2_image_zoom_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_image_zoom_f=0.4',
  'output_path': 'output_CNMC_2_image_zoom_f=0.4'
  }
data['CNMC_2_image_zoom_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_image_zoom_f=0.5',
  'output_path': 'output_CNMC_2_image_zoom_f=0.5'
  }
data['CNMC_2_image_zoom_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_image_zoom_f=0.6',
  'output_path': 'output_CNMC_2_image_zoom_f=0.6'
  }
data['CNMC_2_image_zoom_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_image_zoom_f=0.7',
  'output_path': 'output_CNMC_2_image_zoom_f=0.7'
  }
data['CNMC_2_image_zoom_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_image_zoom_f=0.8',
  'output_path': 'output_CNMC_2_image_zoom_f=0.8'
  }
data['CNMC_2_image_zoom_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_image_zoom_f=0.9',
  'output_path': 'output_CNMC_2_image_zoom_f=0.9'
  }
data['CNMC_2_image_zoom_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_image_zoom_f=1.0',
  'output_path': 'output_CNMC_2_image_zoom_f=1.0'
  }

data['CNMC_2_add_noise_gaussian_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_gaussian_f=0.1',
  'output_path': 'output_CNMC_2_add_noise_gaussian_f=0.1'
  }
data['CNMC_2_add_noise_gaussian_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_gaussian_f=0.2',
  'output_path': 'output_CNMC_2_add_noise_gaussian_f=0.2'
  }
data['CNMC_2_add_noise_gaussian_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_gaussian_f=0.3',
  'output_path': 'output_CNMC_2_add_noise_gaussian_f=0.3'
  }
data['CNMC_2_add_noise_gaussian_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_gaussian_f=0.4',
  'output_path': 'output_CNMC_2_add_noise_gaussian_f=0.4'
  }
data['CNMC_2_add_noise_gaussian_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_gaussian_f=0.5',
  'output_path': 'output_CNMC_2_add_noise_gaussian_f=0.5'
  }
data['CNMC_2_add_noise_gaussian_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_gaussian_f=0.6',
  'output_path': 'output_CNMC_2_add_noise_gaussian_f=0.6'
  }
data['CNMC_2_add_noise_gaussian_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_gaussian_f=0.7',
  'output_path': 'output_CNMC_2_add_noise_gaussian_f=0.7'
  }
data['CNMC_2_add_noise_gaussian_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_gaussian_f=0.8',
  'output_path': 'output_CNMC_2_add_noise_gaussian_f=0.8'
  }
data['CNMC_2_add_noise_gaussian_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_gaussian_f=0.9',
  'output_path': 'output_CNMC_2_add_noise_gaussian_f=0.9'
  }
data['CNMC_2_add_noise_gaussian_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_gaussian_f=1.0',
  'output_path': 'output_CNMC_2_add_noise_gaussian_f=1.0'
  }

data['CNMC_2_add_noise_poisson_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_poisson_f=0.1',
  'output_path': 'output_CNMC_2_add_noise_poisson_f=0.1'
  }
data['CNMC_2_add_noise_poisson_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_poisson_f=0.2',
  'output_path': 'output_CNMC_2_add_noise_poisson_f=0.2'
  }
data['CNMC_2_add_noise_poisson_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_poisson_f=0.3',
  'output_path': 'output_CNMC_2_add_noise_poisson_f=0.3'
  }
data['CNMC_2_add_noise_poisson_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_poisson_f=0.4',
  'output_path': 'output_CNMC_2_add_noise_poisson_f=0.4'
  }
data['CNMC_2_add_noise_poisson_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_poisson_f=0.5',
  'output_path': 'output_CNMC_2_add_noise_poisson_f=0.5'
  }
data['CNMC_2_add_noise_poisson_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_poisson_f=0.6',
  'output_path': 'output_CNMC_2_add_noise_poisson_f=0.6'
  }
data['CNMC_2_add_noise_poisson_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_poisson_f=0.7',
  'output_path': 'output_CNMC_2_add_noise_poisson_f=0.7'
  }
data['CNMC_2_add_noise_poisson_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_poisson_f=0.8',
  'output_path': 'output_CNMC_2_add_noise_poisson_f=0.8'
  }
data['CNMC_2_add_noise_poisson_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_poisson_f=0.9',
  'output_path': 'output_CNMC_2_add_noise_poisson_f=0.9'
  }
data['CNMC_2_add_noise_poisson_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_poisson_f=1.0',
  'output_path': 'output_CNMC_2_add_noise_poisson_f=1.0'
  }

data['CNMC_2_add_noise_salt_and_pepper_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_salt_and_pepper_f=0.1',
  'output_path': 'output_CNMC_2_add_noise_salt_and_pepper_f=0.1'
  }
data['CNMC_2_add_noise_salt_and_pepper_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_salt_and_pepper_f=0.2',
  'output_path': 'output_CNMC_2_add_noise_salt_and_pepper_f=0.2'
  }
data['CNMC_2_add_noise_salt_and_pepper_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_salt_and_pepper_f=0.3',
  'output_path': 'output_CNMC_2_add_noise_salt_and_pepper_f=0.3'
  }
data['CNMC_2_add_noise_salt_and_pepper_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_salt_and_pepper_f=0.4',
  'output_path': 'output_CNMC_2_add_noise_salt_and_pepper_f=0.4'
  }
data['CNMC_2_add_noise_salt_and_pepper_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_salt_and_pepper_f=0.5',
  'output_path': 'output_CNMC_2_add_noise_salt_and_pepper_f=0.5'
  }
data['CNMC_2_add_noise_salt_and_pepper_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_salt_and_pepper_f=0.6',
  'output_path': 'output_CNMC_2_add_noise_salt_and_pepper_f=0.6'
  }
data['CNMC_2_add_noise_salt_and_pepper_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_salt_and_pepper_f=0.7',
  'output_path': 'output_CNMC_2_add_noise_salt_and_pepper_f=0.7'
  }
data['CNMC_2_add_noise_salt_and_pepper_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_salt_and_pepper_f=0.8',
  'output_path': 'output_CNMC_2_add_noise_salt_and_pepper_f=0.8'
  }
data['CNMC_2_add_noise_salt_and_pepper_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_salt_and_pepper_f=0.9',
  'output_path': 'output_CNMC_2_add_noise_salt_and_pepper_f=0.9'
  }
data['CNMC_2_add_noise_salt_and_pepper_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_salt_and_pepper_f=1.0',
  'output_path': 'output_CNMC_2_add_noise_salt_and_pepper_f=1.0'
  }

data['CNMC_2_add_noise_speckle_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_speckle_f=0.1',
  'output_path': 'output_CNMC_2_add_noise_speckle_f=0.1'
  }
data['CNMC_2_add_noise_speckle_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_speckle_f=0.2',
  'output_path': 'output_CNMC_2_add_noise_speckle_f=0.2'
  }
data['CNMC_2_add_noise_speckle_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_speckle_f=0.3',
  'output_path': 'output_CNMC_2_add_noise_speckle_f=0.3'
  }
data['CNMC_2_add_noise_speckle_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_speckle_f=0.4',
  'output_path': 'output_CNMC_2_add_noise_speckle_f=0.4'
  }
data['CNMC_2_add_noise_speckle_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_speckle_f=0.5',
  'output_path': 'output_CNMC_2_add_noise_speckle_f=0.5'
  }
data['CNMC_2_add_noise_speckle_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_speckle_f=0.6',
  'output_path': 'output_CNMC_2_add_noise_speckle_f=0.6'
  }
data['CNMC_2_add_noise_speckle_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_speckle_f=0.7',
  'output_path': 'output_CNMC_2_add_noise_speckle_f=0.7'
  }
data['CNMC_2_add_noise_speckle_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_speckle_f=0.8',
  'output_path': 'output_CNMC_2_add_noise_speckle_f=0.8'
  }
data['CNMC_2_add_noise_speckle_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_speckle_f=0.9',
  'output_path': 'output_CNMC_2_add_noise_speckle_f=0.9'
  }
data['CNMC_2_add_noise_speckle_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_add_noise_speckle_f=1.0',
  'output_path': 'output_CNMC_2_add_noise_speckle_f=1.0'
  }

data['CNMC_2_imbalance_classes_f=0.1'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_imbalance_classes_f=0.1',
'output_path': 'output_CNMC_2_imbalance_classes_f=0.1'
}
data['CNMC_2_imbalance_classes_f=0.2'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_imbalance_classes_f=0.2',
'output_path': 'output_CNMC_2_imbalance_classes_f=0.2'
}
data['CNMC_2_imbalance_classes_f=0.3'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_imbalance_classes_f=0.3',
'output_path': 'output_CNMC_2_imbalance_classes_f=0.3'
}
data['CNMC_2_imbalance_classes_f=0.4'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_imbalance_classes_f=0.4',
'output_path': 'output_CNMC_2_imbalance_classes_f=0.4'
}
data['CNMC_2_imbalance_classes_f=0.5'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_imbalance_classes_f=0.5',
'output_path': 'output_CNMC_2_imbalance_classes_f=0.5'
}
data['CNMC_2_imbalance_classes_f=0.6'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_imbalance_classes_f=0.6',
'output_path': 'output_CNMC_2_imbalance_classes_f=0.6'
}
data['CNMC_2_imbalance_classes_f=0.7'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_imbalance_classes_f=0.7',
'output_path': 'output_CNMC_2_imbalance_classes_f=0.7'
}
data['CNMC_2_imbalance_classes_f=0.8'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_imbalance_classes_f=0.8',
'output_path': 'output_CNMC_2_imbalance_classes_f=0.8'
}
data['CNMC_2_imbalance_classes_f=0.9'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_imbalance_classes_f=0.9',
'output_path': 'output_CNMC_2_imbalance_classes_f=0.9'
}
data['CNMC_2_imbalance_classes_f=1.0'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_imbalance_classes_f=1.0',
'output_path': 'output_CNMC_2_imbalance_classes_f=1.0'
}

data['CNMC_2_grayscale_f=0.1'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_grayscale_f=0.1',
'output_path': 'output_CNMC_2_grayscale_f=0.1'
}
data['CNMC_2_grayscale_f=0.2'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_grayscale_f=0.2',
'output_path': 'output_CNMC_2_grayscale_f=0.2'
}
data['CNMC_2_grayscale_f=0.3'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_grayscale_f=0.3',
'output_path': 'output_CNMC_2_grayscale_f=0.3'
}
data['CNMC_2_grayscale_f=0.4'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_grayscale_f=0.4',
'output_path': 'output_CNMC_2_grayscale_f=0.4'
}
data['CNMC_2_grayscale_f=0.5'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_grayscale_f=0.5',
'output_path': 'output_CNMC_2_grayscale_f=0.5'
}
data['CNMC_2_grayscale_f=0.6'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_grayscale_f=0.6',
'output_path': 'output_CNMC_2_grayscale_f=0.6'
}
data['CNMC_2_grayscale_f=0.7'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_grayscale_f=0.7',
'output_path': 'output_CNMC_2_grayscale_f=0.7'
}
data['CNMC_2_grayscale_f=0.8'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_grayscale_f=0.8',
'output_path': 'output_CNMC_2_grayscale_f=0.8'
}
data['CNMC_2_grayscale_f=0.9'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_grayscale_f=0.9',
'output_path': 'output_CNMC_2_grayscale_f=0.9'
}
data['CNMC_2_grayscale_f=1.0'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_grayscale_f=1.0',
'output_path': 'output_CNMC_2_grayscale_f=1.0'
}

data['CNMC_2_hsv_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_hsv_f=0.1',
'output_path': 'output_CNMC_2_hsv_f=0.1'
}
data['CNMC_2_hsv_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_hsv_f=0.2',
'output_path': 'output_CNMC_2_hsv_f=0.2'
}
data['CNMC_2_hsv_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_hsv_f=0.3',
'output_path': 'output_CNMC_2_hsv_f=0.3'
}
data['CNMC_2_hsv_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_hsv_f=0.4',
'output_path': 'output_CNMC_2_hsv_f=0.4'
}
data['CNMC_2_hsv_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_hsv_f=0.5',
'output_path': 'output_CNMC_2_hsv_f=0.5'
}
data['CNMC_2_hsv_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_hsv_f=0.6',
'output_path': 'output_CNMC_2_hsv_f=0.6'
}
data['CNMC_2_hsv_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_hsv_f=0.7',
'output_path': 'output_CNMC_2_hsv_f=0.7'
}
data['CNMC_2_hsv_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_hsv_f=0.8',
'output_path': 'output_CNMC_2_hsv_f=0.8'
}
data['CNMC_2_hsv_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_hsv_f=0.9',
'output_path': 'output_CNMC_2_hsv_f=0.9'
}
data['CNMC_2_hsv_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_hsv_f=1.0',
'output_path': 'output_CNMC_2_hsv_f=1.0'
}
data['CNMC_2_blur_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_blur_f=0.1',
'output_path': 'output_CNMC_2_blur_f=0.1'
}
data['CNMC_2_blur_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_blur_f=0.2',
'output_path': 'output_CNMC_2_blur_f=0.2'
}
data['CNMC_2_blur_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_blur_f=0.3',
'output_path': 'output_CNMC_2_blur_f=0.3'
}
data['CNMC_2_blur_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_blur_f=0.4',
'output_path': 'output_CNMC_2_blur_f=0.4'
}
data['CNMC_2_blur_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_blur_f=0.5',
'output_path': 'output_CNMC_2_blur_f=0.5'
}
data['CNMC_2_blur_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_blur_f=0.6',
'output_path': 'output_CNMC_2_blur_f=0.6'
}
data['CNMC_2_blur_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_blur_f=0.7',
'output_path': 'output_CNMC_2_blur_f=0.7'
}
data['CNMC_2_blur_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_blur_f=0.8',
'output_path': 'output_CNMC_2_blur_f=0.8'
}
data['CNMC_2_blur_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_blur_f=0.9',
'output_path': 'output_CNMC_2_blur_f=0.9'
}
data['CNMC_2_blur_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_blur_f=1.0',
'output_path': 'output_CNMC_2_blur_f=1.0'
}

data['CNMC_2_small_random_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_small_random_f=0.1',
'output_path': 'output_CNMC_2_small_random_f=0.1'
}
data['CNMC_2_small_random_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_small_random_f=0.2',
'output_path': 'output_CNMC_2_small_random_f=0.2'
}
data['CNMC_2_small_random_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_small_random_f=0.3',
'output_path': 'output_CNMC_2_small_random_f=0.3'
}
data['CNMC_2_small_random_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_small_random_f=0.4',
'output_path': 'output_CNMC_2_small_random_f=0.4'
}
data['CNMC_2_small_random_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_small_random_f=0.5',
'output_path': 'output_CNMC_2_small_random_f=0.5'
}
data['CNMC_2_small_random_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_small_random_f=0.6',
'output_path': 'output_CNMC_2_small_random_f=0.6'
}
data['CNMC_2_small_random_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_small_random_f=0.7',
'output_path': 'output_CNMC_2_small_random_f=0.7'
}
data['CNMC_2_small_random_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_small_random_f=0.8',
'output_path': 'output_CNMC_2_small_random_f=0.8'
}
data['CNMC_2_small_random_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_small_random_f=0.9',
'output_path': 'output_CNMC_2_small_random_f=0.9'
}
data['CNMC_2_small_random_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_small_random_f=1.0',
'output_path': 'output_CNMC_2_small_random_f=1.0'
}

data['CNMC_2_small_easy_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_small_easy_f=0.1',
  'output_path': 'output_CNMC_2_small_easy_f=0.1'
  }
data['CNMC_2_small_easy_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_small_easy_f=0.2',
  'output_path': 'output_CNMC_2_small_easy_f=0.2'
  }
data['CNMC_2_small_easy_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_small_easy_f=0.3',
  'output_path': 'output_CNMC_2_small_easy_f=0.3'
  }
data['CNMC_2_small_easy_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_small_easy_f=0.4',
  'output_path': 'output_CNMC_2_small_easy_f=0.4'
  }
data['CNMC_2_small_easy_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_small_easy_f=0.5',
  'output_path': 'output_CNMC_2_small_easy_f=0.5'
  }
data['CNMC_2_small_easy_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_small_easy_f=0.6',
  'output_path': 'output_CNMC_2_small_easy_f=0.6'
  }
data['CNMC_2_small_easy_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_small_easy_f=0.7',
  'output_path': 'output_CNMC_2_small_easy_f=0.7'
  }
data['CNMC_2_small_easy_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_small_easy_f=0.8',
  'output_path': 'output_CNMC_2_small_easy_f=0.8'
  }
data['CNMC_2_small_easy_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_small_easy_f=0.9',
  'output_path': 'output_CNMC_2_small_easy_f=0.9'
  }
data['CNMC_2_small_easy_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_small_easy_f=1.0',
  'output_path': 'output_CNMC_2_small_easy_f=1.0'
  }

data['CNMC_2_small_hard_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_small_hard_f=0.1',
  'output_path': 'output_CNMC_2_small_hard_f=0.1'
  }
data['CNMC_2_small_hard_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_small_hard_f=0.2',
  'output_path': 'output_CNMC_2_small_hard_f=0.2'
  }
data['CNMC_2_small_hard_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_small_hard_f=0.3',
  'output_path': 'output_CNMC_2_small_hard_f=0.3'
  }
data['CNMC_2_small_hard_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_small_hard_f=0.4',
  'output_path': 'output_CNMC_2_small_hard_f=0.4'
  }
data['CNMC_2_small_hard_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_small_hard_f=0.5',
  'output_path': 'output_CNMC_2_small_hard_f=0.5'
  }
data['CNMC_2_small_hard_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_small_hard_f=0.6',
  'output_path': 'output_CNMC_2_small_hard_f=0.6'
  }
data['CNMC_2_small_hard_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_small_hard_f=0.7',
  'output_path': 'output_CNMC_2_small_hard_f=0.7'
  }
data['CNMC_2_small_hard_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_small_hard_f=0.8',
  'output_path': 'output_CNMC_2_small_hard_f=0.8'
  }
data['CNMC_2_small_hard_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_small_hard_f=0.9',
  'output_path': 'output_CNMC_2_small_hard_f=0.9'
  }
data['CNMC_2_small_hard_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_2',
  'dataset_path': 'dataset_CNMC_2_small_hard_f=1.0',
  'output_path': 'output_CNMC_2_small_hard_f=1.0'
  }

data['CNMC_2_small_clusters_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_small_clusters_f=0.1',
'output_path': 'output_CNMC_2_small_clusters_f=0.1'
}
data['CNMC_2_small_clusters_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_small_clusters_f=0.2',
'output_path': 'output_CNMC_2_small_clusters_f=0.2'
}
data['CNMC_2_small_clusters_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_small_clusters_f=0.3',
'output_path': 'output_CNMC_2_small_clusters_f=0.3'
}
data['CNMC_2_small_clusters_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_small_clusters_f=0.4',
'output_path': 'output_CNMC_2_small_clusters_f=0.4'
}
data['CNMC_2_small_clusters_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_small_clusters_f=0.5',
'output_path': 'output_CNMC_2_small_clusters_f=0.5'
}
data['CNMC_2_small_clusters_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_small_clusters_f=0.6',
'output_path': 'output_CNMC_2_small_clusters_f=0.6'
}
data['CNMC_2_small_clusters_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_small_clusters_f=0.7',
'output_path': 'output_CNMC_2_small_clusters_f=0.7'
}
data['CNMC_2_small_clusters_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_small_clusters_f=0.8',
'output_path': 'output_CNMC_2_small_clusters_f=0.8'
}
data['CNMC_2_small_clusters_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_small_clusters_f=0.9',
'output_path': 'output_CNMC_2_small_clusters_f=0.9'
}
data['CNMC_2_small_clusters_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_2',
'dataset_path': 'dataset_CNMC_2_small_clusters_f=1.0',
'output_path': 'output_CNMC_2_small_clusters_f=1.0'
}

data['CNMC_3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3',
  'output_path': 'output_CNMC_3'
  }
data['CNMC_3_image_rot_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_image_rot_f=0.1',
  'output_path': 'output_CNMC_3_image_rot_f=0.1'
  }
data['CNMC_3_image_rot_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_image_rot_f=0.2',
  'output_path': 'output_CNMC_3_image_rot_f=0.2'
  }
data['CNMC_3_image_rot_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_image_rot_f=0.3',
  'output_path': 'output_CNMC_3_image_rot_f=0.3'
  }
data['CNMC_3_image_rot_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_image_rot_f=0.4',
  'output_path': 'output_CNMC_3_image_rot_f=0.4'
  }
data['CNMC_3_image_rot_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_image_rot_f=0.5',
  'output_path': 'output_CNMC_3_image_rot_f=0.5'
  }
data['CNMC_3_image_rot_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_image_rot_f=0.6',
  'output_path': 'output_CNMC_3_image_rot_f=0.6'
  }
data['CNMC_3_image_rot_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_image_rot_f=0.7',
  'output_path': 'output_CNMC_3_image_rot_f=0.7'
  }
data['CNMC_3_image_rot_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_image_rot_f=0.8',
  'output_path': 'output_CNMC_3_image_rot_f=0.8'
  }
data['CNMC_3_image_rot_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_image_rot_f=0.9',
  'output_path': 'output_CNMC_3_image_rot_f=0.9'
  }
data['CNMC_3_image_rot_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_image_rot_f=1.0',
  'output_path': 'output_CNMC_3_image_rot_f=1.0'
  }

data['CNMC_3_image_translation_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_image_translation_f=0.1',
  'output_path': 'output_CNMC_3_image_translation_f=0.1'
  }
data['CNMC_3_image_translation_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_image_translation_f=0.2',
  'output_path': 'output_CNMC_3_image_translation_f=0.2'
  }
data['CNMC_3_image_translation_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_image_translation_f=0.3',
  'output_path': 'output_CNMC_3_image_translation_f=0.3'
  }
data['CNMC_3_image_translation_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_image_translation_f=0.4',
  'output_path': 'output_CNMC_3_image_translation_f=0.4'
  }
data['CNMC_3_image_translation_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_image_translation_f=0.5',
  'output_path': 'output_CNMC_3_image_translation_f=0.5'
  }
data['CNMC_3_image_translation_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_image_translation_f=0.6',
  'output_path': 'output_CNMC_3_image_translation_f=0.6'
  }
data['CNMC_3_image_translation_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_image_translation_f=0.7',
  'output_path': 'output_CNMC_3_image_translation_f=0.7'
  }
data['CNMC_3_image_translation_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_image_translation_f=0.8',
  'output_path': 'output_CNMC_3_image_translation_f=0.8'
  }
data['CNMC_3_image_translation_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_image_translation_f=0.9',
  'output_path': 'output_CNMC_3_image_translation_f=0.9'
  }
data['CNMC_3_image_translation_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_image_translation_f=1.0',
  'output_path': 'output_CNMC_3_image_translation_f=1.0'
  }

data['CNMC_3_image_zoom_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_image_zoom_f=0.1',
  'output_path': 'output_CNMC_3_image_zoom_f=0.1'
  }
data['CNMC_3_image_zoom_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_image_zoom_f=0.2',
  'output_path': 'output_CNMC_3_image_zoom_f=0.2'
  }
data['CNMC_3_image_zoom_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_image_zoom_f=0.3',
  'output_path': 'output_CNMC_3_image_zoom_f=0.3'
  }
data['CNMC_3_image_zoom_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_image_zoom_f=0.4',
  'output_path': 'output_CNMC_3_image_zoom_f=0.4'
  }
data['CNMC_3_image_zoom_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_image_zoom_f=0.5',
  'output_path': 'output_CNMC_3_image_zoom_f=0.5'
  }
data['CNMC_3_image_zoom_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_image_zoom_f=0.6',
  'output_path': 'output_CNMC_3_image_zoom_f=0.6'
  }
data['CNMC_3_image_zoom_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_image_zoom_f=0.7',
  'output_path': 'output_CNMC_3_image_zoom_f=0.7'
  }
data['CNMC_3_image_zoom_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_image_zoom_f=0.8',
  'output_path': 'output_CNMC_3_image_zoom_f=0.8'
  }
data['CNMC_3_image_zoom_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_image_zoom_f=0.9',
  'output_path': 'output_CNMC_3_image_zoom_f=0.9'
  }
data['CNMC_3_image_zoom_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_image_zoom_f=1.0',
  'output_path': 'output_CNMC_3_image_zoom_f=1.0'
  }

data['CNMC_3_add_noise_gaussian_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_gaussian_f=0.1',
  'output_path': 'output_CNMC_3_add_noise_gaussian_f=0.1'
  }
data['CNMC_3_add_noise_gaussian_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_gaussian_f=0.2',
  'output_path': 'output_CNMC_3_add_noise_gaussian_f=0.2'
  }
data['CNMC_3_add_noise_gaussian_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_gaussian_f=0.3',
  'output_path': 'output_CNMC_3_add_noise_gaussian_f=0.3'
  }
data['CNMC_3_add_noise_gaussian_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_gaussian_f=0.4',
  'output_path': 'output_CNMC_3_add_noise_gaussian_f=0.4'
  }
data['CNMC_3_add_noise_gaussian_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_gaussian_f=0.5',
  'output_path': 'output_CNMC_3_add_noise_gaussian_f=0.5'
  }
data['CNMC_3_add_noise_gaussian_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_gaussian_f=0.6',
  'output_path': 'output_CNMC_3_add_noise_gaussian_f=0.6'
  }
data['CNMC_3_add_noise_gaussian_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_gaussian_f=0.7',
  'output_path': 'output_CNMC_3_add_noise_gaussian_f=0.7'
  }
data['CNMC_3_add_noise_gaussian_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_gaussian_f=0.8',
  'output_path': 'output_CNMC_3_add_noise_gaussian_f=0.8'
  }
data['CNMC_3_add_noise_gaussian_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_gaussian_f=0.9',
  'output_path': 'output_CNMC_3_add_noise_gaussian_f=0.9'
  }
data['CNMC_3_add_noise_gaussian_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_gaussian_f=1.0',
  'output_path': 'output_CNMC_3_add_noise_gaussian_f=1.0'
  }

data['CNMC_3_add_noise_poisson_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_poisson_f=0.1',
  'output_path': 'output_CNMC_3_add_noise_poisson_f=0.1'
  }
data['CNMC_3_add_noise_poisson_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_poisson_f=0.2',
  'output_path': 'output_CNMC_3_add_noise_poisson_f=0.2'
  }
data['CNMC_3_add_noise_poisson_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_poisson_f=0.3',
  'output_path': 'output_CNMC_3_add_noise_poisson_f=0.3'
  }
data['CNMC_3_add_noise_poisson_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_poisson_f=0.4',
  'output_path': 'output_CNMC_3_add_noise_poisson_f=0.4'
  }
data['CNMC_3_add_noise_poisson_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_poisson_f=0.5',
  'output_path': 'output_CNMC_3_add_noise_poisson_f=0.5'
  }
data['CNMC_3_add_noise_poisson_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_poisson_f=0.6',
  'output_path': 'output_CNMC_3_add_noise_poisson_f=0.6'
  }
data['CNMC_3_add_noise_poisson_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_poisson_f=0.7',
  'output_path': 'output_CNMC_3_add_noise_poisson_f=0.7'
  }
data['CNMC_3_add_noise_poisson_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_poisson_f=0.8',
  'output_path': 'output_CNMC_3_add_noise_poisson_f=0.8'
  }
data['CNMC_3_add_noise_poisson_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_poisson_f=0.9',
  'output_path': 'output_CNMC_3_add_noise_poisson_f=0.9'
  }
data['CNMC_3_add_noise_poisson_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_poisson_f=1.0',
  'output_path': 'output_CNMC_3_add_noise_poisson_f=1.0'
  }

data['CNMC_3_add_noise_salt_and_pepper_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_salt_and_pepper_f=0.1',
  'output_path': 'output_CNMC_3_add_noise_salt_and_pepper_f=0.1'
  }
data['CNMC_3_add_noise_salt_and_pepper_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_salt_and_pepper_f=0.2',
  'output_path': 'output_CNMC_3_add_noise_salt_and_pepper_f=0.2'
  }
data['CNMC_3_add_noise_salt_and_pepper_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_salt_and_pepper_f=0.3',
  'output_path': 'output_CNMC_3_add_noise_salt_and_pepper_f=0.3'
  }
data['CNMC_3_add_noise_salt_and_pepper_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_salt_and_pepper_f=0.4',
  'output_path': 'output_CNMC_3_add_noise_salt_and_pepper_f=0.4'
  }
data['CNMC_3_add_noise_salt_and_pepper_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_salt_and_pepper_f=0.5',
  'output_path': 'output_CNMC_3_add_noise_salt_and_pepper_f=0.5'
  }
data['CNMC_3_add_noise_salt_and_pepper_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_salt_and_pepper_f=0.6',
  'output_path': 'output_CNMC_3_add_noise_salt_and_pepper_f=0.6'
  }
data['CNMC_3_add_noise_salt_and_pepper_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_salt_and_pepper_f=0.7',
  'output_path': 'output_CNMC_3_add_noise_salt_and_pepper_f=0.7'
  }
data['CNMC_3_add_noise_salt_and_pepper_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_salt_and_pepper_f=0.8',
  'output_path': 'output_CNMC_3_add_noise_salt_and_pepper_f=0.8'
  }
data['CNMC_3_add_noise_salt_and_pepper_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_salt_and_pepper_f=0.9',
  'output_path': 'output_CNMC_3_add_noise_salt_and_pepper_f=0.9'
  }
data['CNMC_3_add_noise_salt_and_pepper_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_salt_and_pepper_f=1.0',
  'output_path': 'output_CNMC_3_add_noise_salt_and_pepper_f=1.0'
  }

data['CNMC_3_add_noise_speckle_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_speckle_f=0.1',
  'output_path': 'output_CNMC_3_add_noise_speckle_f=0.1'
  }
data['CNMC_3_add_noise_speckle_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_speckle_f=0.2',
  'output_path': 'output_CNMC_3_add_noise_speckle_f=0.2'
  }
data['CNMC_3_add_noise_speckle_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_speckle_f=0.3',
  'output_path': 'output_CNMC_3_add_noise_speckle_f=0.3'
  }
data['CNMC_3_add_noise_speckle_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_speckle_f=0.4',
  'output_path': 'output_CNMC_3_add_noise_speckle_f=0.4'
  }
data['CNMC_3_add_noise_speckle_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_speckle_f=0.5',
  'output_path': 'output_CNMC_3_add_noise_speckle_f=0.5'
  }
data['CNMC_3_add_noise_speckle_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_speckle_f=0.6',
  'output_path': 'output_CNMC_3_add_noise_speckle_f=0.6'
  }
data['CNMC_3_add_noise_speckle_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_speckle_f=0.7',
  'output_path': 'output_CNMC_3_add_noise_speckle_f=0.7'
  }
data['CNMC_3_add_noise_speckle_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_speckle_f=0.8',
  'output_path': 'output_CNMC_3_add_noise_speckle_f=0.8'
  }
data['CNMC_3_add_noise_speckle_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_speckle_f=0.9',
  'output_path': 'output_CNMC_3_add_noise_speckle_f=0.9'
  }
data['CNMC_3_add_noise_speckle_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_add_noise_speckle_f=1.0',
  'output_path': 'output_CNMC_3_add_noise_speckle_f=1.0'
  }

data['CNMC_3_imbalance_classes_f=0.1'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_imbalance_classes_f=0.1',
'output_path': 'output_CNMC_3_imbalance_classes_f=0.1'
}
data['CNMC_3_imbalance_classes_f=0.2'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_imbalance_classes_f=0.2',
'output_path': 'output_CNMC_3_imbalance_classes_f=0.2'
}
data['CNMC_3_imbalance_classes_f=0.3'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_imbalance_classes_f=0.3',
'output_path': 'output_CNMC_3_imbalance_classes_f=0.3'
}
data['CNMC_3_imbalance_classes_f=0.4'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_imbalance_classes_f=0.4',
'output_path': 'output_CNMC_3_imbalance_classes_f=0.4'
}
data['CNMC_3_imbalance_classes_f=0.5'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_imbalance_classes_f=0.5',
'output_path': 'output_CNMC_3_imbalance_classes_f=0.5'
}
data['CNMC_3_imbalance_classes_f=0.6'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_imbalance_classes_f=0.6',
'output_path': 'output_CNMC_3_imbalance_classes_f=0.6'
}
data['CNMC_3_imbalance_classes_f=0.7'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_imbalance_classes_f=0.7',
'output_path': 'output_CNMC_3_imbalance_classes_f=0.7'
}
data['CNMC_3_imbalance_classes_f=0.8'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_imbalance_classes_f=0.8',
'output_path': 'output_CNMC_3_imbalance_classes_f=0.8'
}
data['CNMC_3_imbalance_classes_f=0.9'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_imbalance_classes_f=0.9',
'output_path': 'output_CNMC_3_imbalance_classes_f=0.9'
}
data['CNMC_3_imbalance_classes_f=1.0'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_imbalance_classes_f=1.0',
'output_path': 'output_CNMC_3_imbalance_classes_f=1.0'
}

data['CNMC_3_grayscale_f=0.1'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_grayscale_f=0.1',
'output_path': 'output_CNMC_3_grayscale_f=0.1'
}
data['CNMC_3_grayscale_f=0.2'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_grayscale_f=0.2',
'output_path': 'output_CNMC_3_grayscale_f=0.2'
}
data['CNMC_3_grayscale_f=0.3'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_grayscale_f=0.3',
'output_path': 'output_CNMC_3_grayscale_f=0.3'
}
data['CNMC_3_grayscale_f=0.4'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_grayscale_f=0.4',
'output_path': 'output_CNMC_3_grayscale_f=0.4'
}
data['CNMC_3_grayscale_f=0.5'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_grayscale_f=0.5',
'output_path': 'output_CNMC_3_grayscale_f=0.5'
}
data['CNMC_3_grayscale_f=0.6'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_grayscale_f=0.6',
'output_path': 'output_CNMC_3_grayscale_f=0.6'
}
data['CNMC_3_grayscale_f=0.7'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_grayscale_f=0.7',
'output_path': 'output_CNMC_3_grayscale_f=0.7'
}
data['CNMC_3_grayscale_f=0.8'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_grayscale_f=0.8',
'output_path': 'output_CNMC_3_grayscale_f=0.8'
}
data['CNMC_3_grayscale_f=0.9'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_grayscale_f=0.9',
'output_path': 'output_CNMC_3_grayscale_f=0.9'
}
data['CNMC_3_grayscale_f=1.0'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_grayscale_f=1.0',
'output_path': 'output_CNMC_3_grayscale_f=1.0'
}
data['CNMC_3_hsv_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_hsv_f=0.1',
'output_path': 'output_CNMC_3_hsv_f=0.1'
}
data['CNMC_3_hsv_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_hsv_f=0.2',
'output_path': 'output_CNMC_3_hsv_f=0.2'
}
data['CNMC_3_hsv_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_hsv_f=0.3',
'output_path': 'output_CNMC_3_hsv_f=0.3'
}
data['CNMC_3_hsv_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_hsv_f=0.4',
'output_path': 'output_CNMC_3_hsv_f=0.4'
}
data['CNMC_3_hsv_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_hsv_f=0.5',
'output_path': 'output_CNMC_3_hsv_f=0.5'
}
data['CNMC_3_hsv_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_hsv_f=0.6',
'output_path': 'output_CNMC_3_hsv_f=0.6'
}
data['CNMC_3_hsv_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_hsv_f=0.7',
'output_path': 'output_CNMC_3_hsv_f=0.7'
}
data['CNMC_3_hsv_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_hsv_f=0.8',
'output_path': 'output_CNMC_3_hsv_f=0.8'
}
data['CNMC_3_hsv_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_hsv_f=0.9',
'output_path': 'output_CNMC_3_hsv_f=0.9'
}
data['CNMC_3_hsv_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_hsv_f=1.0',
'output_path': 'output_CNMC_3_hsv_f=1.0'
}
data['CNMC_3_blur_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_blur_f=0.1',
'output_path': 'output_CNMC_3_blur_f=0.1'
}
data['CNMC_3_blur_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_blur_f=0.2',
'output_path': 'output_CNMC_3_blur_f=0.2'
}
data['CNMC_3_blur_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_blur_f=0.3',
'output_path': 'output_CNMC_3_blur_f=0.3'
}
data['CNMC_3_blur_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_blur_f=0.4',
'output_path': 'output_CNMC_3_blur_f=0.4'
}
data['CNMC_3_blur_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_blur_f=0.5',
'output_path': 'output_CNMC_3_blur_f=0.5'
}
data['CNMC_3_blur_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_blur_f=0.6',
'output_path': 'output_CNMC_3_blur_f=0.6'
}
data['CNMC_3_blur_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_blur_f=0.7',
'output_path': 'output_CNMC_3_blur_f=0.7'
}
data['CNMC_3_blur_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_blur_f=0.8',
'output_path': 'output_CNMC_3_blur_f=0.8'
}
data['CNMC_3_blur_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_blur_f=0.9',
'output_path': 'output_CNMC_3_blur_f=0.9'
}
data['CNMC_3_blur_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_blur_f=1.0',
'output_path': 'output_CNMC_3_blur_f=1.0'
}

data['CNMC_3_small_random_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_small_random_f=0.1',
'output_path': 'output_CNMC_3_small_random_f=0.1'
}
data['CNMC_3_small_random_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_small_random_f=0.2',
'output_path': 'output_CNMC_3_small_random_f=0.2'
}
data['CNMC_3_small_random_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_small_random_f=0.3',
'output_path': 'output_CNMC_3_small_random_f=0.3'
}
data['CNMC_3_small_random_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_small_random_f=0.4',
'output_path': 'output_CNMC_3_small_random_f=0.4'
}
data['CNMC_3_small_random_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_small_random_f=0.5',
'output_path': 'output_CNMC_3_small_random_f=0.5'
}
data['CNMC_3_small_random_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_small_random_f=0.6',
'output_path': 'output_CNMC_3_small_random_f=0.6'
}
data['CNMC_3_small_random_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_small_random_f=0.7',
'output_path': 'output_CNMC_3_small_random_f=0.7'
}
data['CNMC_3_small_random_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_small_random_f=0.8',
'output_path': 'output_CNMC_3_small_random_f=0.8'
}
data['CNMC_3_small_random_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_small_random_f=0.9',
'output_path': 'output_CNMC_3_small_random_f=0.9'
}
data['CNMC_3_small_random_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_small_random_f=1.0',
'output_path': 'output_CNMC_3_small_random_f=1.0'
}

data['CNMC_3_small_easy_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_small_easy_f=0.1',
  'output_path': 'output_CNMC_3_small_easy_f=0.1'
  }
data['CNMC_3_small_easy_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_small_easy_f=0.2',
  'output_path': 'output_CNMC_3_small_easy_f=0.2'
  }
data['CNMC_3_small_easy_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_small_easy_f=0.3',
  'output_path': 'output_CNMC_3_small_easy_f=0.3'
  }
data['CNMC_3_small_easy_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_small_easy_f=0.4',
  'output_path': 'output_CNMC_3_small_easy_f=0.4'
  }
data['CNMC_3_small_easy_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_small_easy_f=0.5',
  'output_path': 'output_CNMC_3_small_easy_f=0.5'
  }
data['CNMC_3_small_easy_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_small_easy_f=0.6',
  'output_path': 'output_CNMC_3_small_easy_f=0.6'
  }
data['CNMC_3_small_easy_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_small_easy_f=0.7',
  'output_path': 'output_CNMC_3_small_easy_f=0.7'
  }
data['CNMC_3_small_easy_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_small_easy_f=0.8',
  'output_path': 'output_CNMC_3_small_easy_f=0.8'
  }
data['CNMC_3_small_easy_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_small_easy_f=0.9',
  'output_path': 'output_CNMC_3_small_easy_f=0.9'
  }
data['CNMC_3_small_easy_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_small_easy_f=1.0',
  'output_path': 'output_CNMC_3_small_easy_f=1.0'
  }

data['CNMC_3_small_hard_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_small_hard_f=0.1',
  'output_path': 'output_CNMC_3_small_hard_f=0.1'
  }
data['CNMC_3_small_hard_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_small_hard_f=0.2',
  'output_path': 'output_CNMC_3_small_hard_f=0.2'
  }
data['CNMC_3_small_hard_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_small_hard_f=0.3',
  'output_path': 'output_CNMC_3_small_hard_f=0.3'
  }
data['CNMC_3_small_hard_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_small_hard_f=0.4',
  'output_path': 'output_CNMC_3_small_hard_f=0.4'
  }
data['CNMC_3_small_hard_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_small_hard_f=0.5',
  'output_path': 'output_CNMC_3_small_hard_f=0.5'
  }
data['CNMC_3_small_hard_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_small_hard_f=0.6',
  'output_path': 'output_CNMC_3_small_hard_f=0.6'
  }
data['CNMC_3_small_hard_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_small_hard_f=0.7',
  'output_path': 'output_CNMC_3_small_hard_f=0.7'
  }
data['CNMC_3_small_hard_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_small_hard_f=0.8',
  'output_path': 'output_CNMC_3_small_hard_f=0.8'
  }
data['CNMC_3_small_hard_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_small_hard_f=0.9',
  'output_path': 'output_CNMC_3_small_hard_f=0.9'
  }
data['CNMC_3_small_hard_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_3',
  'dataset_path': 'dataset_CNMC_3_small_hard_f=1.0',
  'output_path': 'output_CNMC_3_small_hard_f=1.0'
  }

data['CNMC_3_small_clusters_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_small_clusters_f=0.1',
'output_path': 'output_CNMC_3_small_clusters_f=0.1'
}
data['CNMC_3_small_clusters_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_small_clusters_f=0.2',
'output_path': 'output_CNMC_3_small_clusters_f=0.2'
}
data['CNMC_3_small_clusters_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_small_clusters_f=0.3',
'output_path': 'output_CNMC_3_small_clusters_f=0.3'
}
data['CNMC_3_small_clusters_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_small_clusters_f=0.4',
'output_path': 'output_CNMC_3_small_clusters_f=0.4'
}
data['CNMC_3_small_clusters_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_small_clusters_f=0.5',
'output_path': 'output_CNMC_3_small_clusters_f=0.5'
}
data['CNMC_3_small_clusters_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_small_clusters_f=0.6',
'output_path': 'output_CNMC_3_small_clusters_f=0.6'
}
data['CNMC_3_small_clusters_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_small_clusters_f=0.7',
'output_path': 'output_CNMC_3_small_clusters_f=0.7'
}
data['CNMC_3_small_clusters_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_small_clusters_f=0.8',
'output_path': 'output_CNMC_3_small_clusters_f=0.8'
}
data['CNMC_3_small_clusters_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_small_clusters_f=0.9',
'output_path': 'output_CNMC_3_small_clusters_f=0.9'
}
data['CNMC_3_small_clusters_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_3',
'dataset_path': 'dataset_CNMC_3_small_clusters_f=1.0',
'output_path': 'output_CNMC_3_small_clusters_f=1.0'
}

data['CNMC_4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4',
  'output_path': 'output_CNMC_4'
  }
data['CNMC_4_image_rot_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_image_rot_f=0.1',
  'output_path': 'output_CNMC_4_image_rot_f=0.1'
  }
data['CNMC_4_image_rot_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_image_rot_f=0.2',
  'output_path': 'output_CNMC_4_image_rot_f=0.2'
  }
data['CNMC_4_image_rot_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_image_rot_f=0.3',
  'output_path': 'output_CNMC_4_image_rot_f=0.3'
  }
data['CNMC_4_image_rot_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_image_rot_f=0.4',
  'output_path': 'output_CNMC_4_image_rot_f=0.4'
  }
data['CNMC_4_image_rot_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_image_rot_f=0.5',
  'output_path': 'output_CNMC_4_image_rot_f=0.5'
  }
data['CNMC_4_image_rot_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_image_rot_f=0.6',
  'output_path': 'output_CNMC_4_image_rot_f=0.6'
  }
data['CNMC_4_image_rot_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_image_rot_f=0.7',
  'output_path': 'output_CNMC_4_image_rot_f=0.7'
  }
data['CNMC_4_image_rot_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_image_rot_f=0.8',
  'output_path': 'output_CNMC_4_image_rot_f=0.8'
  }
data['CNMC_4_image_rot_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_image_rot_f=0.9',
  'output_path': 'output_CNMC_4_image_rot_f=0.9'
  }
data['CNMC_4_image_rot_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_image_rot_f=1.0',
  'output_path': 'output_CNMC_4_image_rot_f=1.0'
  }

data['CNMC_4_image_translation_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_image_translation_f=0.1',
  'output_path': 'output_CNMC_4_image_translation_f=0.1'
  }
data['CNMC_4_image_translation_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_image_translation_f=0.2',
  'output_path': 'output_CNMC_4_image_translation_f=0.2'
  }
data['CNMC_4_image_translation_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_image_translation_f=0.3',
  'output_path': 'output_CNMC_4_image_translation_f=0.3'
  }
data['CNMC_4_image_translation_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_image_translation_f=0.4',
  'output_path': 'output_CNMC_4_image_translation_f=0.4'
  }
data['CNMC_4_image_translation_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_image_translation_f=0.5',
  'output_path': 'output_CNMC_4_image_translation_f=0.5'
  }
data['CNMC_4_image_translation_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_image_translation_f=0.6',
  'output_path': 'output_CNMC_4_image_translation_f=0.6'
  }
data['CNMC_4_image_translation_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_image_translation_f=0.7',
  'output_path': 'output_CNMC_4_image_translation_f=0.7'
  }
data['CNMC_4_image_translation_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_image_translation_f=0.8',
  'output_path': 'output_CNMC_4_image_translation_f=0.8'
  }
data['CNMC_4_image_translation_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_image_translation_f=0.9',
  'output_path': 'output_CNMC_4_image_translation_f=0.9'
  }
data['CNMC_4_image_translation_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_image_translation_f=1.0',
  'output_path': 'output_CNMC_4_image_translation_f=1.0'
  }

data['CNMC_4_image_zoom_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_image_zoom_f=0.1',
  'output_path': 'output_CNMC_4_image_zoom_f=0.1'
  }
data['CNMC_4_image_zoom_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_image_zoom_f=0.2',
  'output_path': 'output_CNMC_4_image_zoom_f=0.2'
  }
data['CNMC_4_image_zoom_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_image_zoom_f=0.3',
  'output_path': 'output_CNMC_4_image_zoom_f=0.3'
  }
data['CNMC_4_image_zoom_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_image_zoom_f=0.4',
  'output_path': 'output_CNMC_4_image_zoom_f=0.4'
  }
data['CNMC_4_image_zoom_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_image_zoom_f=0.5',
  'output_path': 'output_CNMC_4_image_zoom_f=0.5'
  }
data['CNMC_4_image_zoom_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_image_zoom_f=0.6',
  'output_path': 'output_CNMC_4_image_zoom_f=0.6'
  }
data['CNMC_4_image_zoom_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_image_zoom_f=0.7',
  'output_path': 'output_CNMC_4_image_zoom_f=0.7'
  }
data['CNMC_4_image_zoom_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_image_zoom_f=0.8',
  'output_path': 'output_CNMC_4_image_zoom_f=0.8'
  }
data['CNMC_4_image_zoom_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_image_zoom_f=0.9',
  'output_path': 'output_CNMC_4_image_zoom_f=0.9'
  }
data['CNMC_4_image_zoom_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_image_zoom_f=1.0',
  'output_path': 'output_CNMC_4_image_zoom_f=1.0'
  }

data['CNMC_4_add_noise_gaussian_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_gaussian_f=0.1',
  'output_path': 'output_CNMC_4_add_noise_gaussian_f=0.1'
  }
data['CNMC_4_add_noise_gaussian_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_gaussian_f=0.2',
  'output_path': 'output_CNMC_4_add_noise_gaussian_f=0.2'
  }
data['CNMC_4_add_noise_gaussian_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_gaussian_f=0.3',
  'output_path': 'output_CNMC_4_add_noise_gaussian_f=0.3'
  }
data['CNMC_4_add_noise_gaussian_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_gaussian_f=0.4',
  'output_path': 'output_CNMC_4_add_noise_gaussian_f=0.4'
  }
data['CNMC_4_add_noise_gaussian_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_gaussian_f=0.5',
  'output_path': 'output_CNMC_4_add_noise_gaussian_f=0.5'
  }
data['CNMC_4_add_noise_gaussian_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_gaussian_f=0.6',
  'output_path': 'output_CNMC_4_add_noise_gaussian_f=0.6'
  }
data['CNMC_4_add_noise_gaussian_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_gaussian_f=0.7',
  'output_path': 'output_CNMC_4_add_noise_gaussian_f=0.7'
  }
data['CNMC_4_add_noise_gaussian_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_gaussian_f=0.8',
  'output_path': 'output_CNMC_4_add_noise_gaussian_f=0.8'
  }
data['CNMC_4_add_noise_gaussian_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_gaussian_f=0.9',
  'output_path': 'output_CNMC_4_add_noise_gaussian_f=0.9'
  }
data['CNMC_4_add_noise_gaussian_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_gaussian_f=1.0',
  'output_path': 'output_CNMC_4_add_noise_gaussian_f=1.0'
  }

data['CNMC_4_add_noise_poisson_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_poisson_f=0.1',
  'output_path': 'output_CNMC_4_add_noise_poisson_f=0.1'
  }
data['CNMC_4_add_noise_poisson_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_poisson_f=0.2',
  'output_path': 'output_CNMC_4_add_noise_poisson_f=0.2'
  }
data['CNMC_4_add_noise_poisson_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_poisson_f=0.3',
  'output_path': 'output_CNMC_4_add_noise_poisson_f=0.3'
  }
data['CNMC_4_add_noise_poisson_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_poisson_f=0.4',
  'output_path': 'output_CNMC_4_add_noise_poisson_f=0.4'
  }
data['CNMC_4_add_noise_poisson_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_poisson_f=0.5',
  'output_path': 'output_CNMC_4_add_noise_poisson_f=0.5'
  }
data['CNMC_4_add_noise_poisson_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_poisson_f=0.6',
  'output_path': 'output_CNMC_4_add_noise_poisson_f=0.6'
  }
data['CNMC_4_add_noise_poisson_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_poisson_f=0.7',
  'output_path': 'output_CNMC_4_add_noise_poisson_f=0.7'
  }
data['CNMC_4_add_noise_poisson_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_poisson_f=0.8',
  'output_path': 'output_CNMC_4_add_noise_poisson_f=0.8'
  }
data['CNMC_4_add_noise_poisson_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_poisson_f=0.9',
  'output_path': 'output_CNMC_4_add_noise_poisson_f=0.9'
  }
data['CNMC_4_add_noise_poisson_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_poisson_f=1.0',
  'output_path': 'output_CNMC_4_add_noise_poisson_f=1.0'
  }

data['CNMC_4_add_noise_salt_and_pepper_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_salt_and_pepper_f=0.1',
  'output_path': 'output_CNMC_4_add_noise_salt_and_pepper_f=0.1'
  }
data['CNMC_4_add_noise_salt_and_pepper_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_salt_and_pepper_f=0.2',
  'output_path': 'output_CNMC_4_add_noise_salt_and_pepper_f=0.2'
  }
data['CNMC_4_add_noise_salt_and_pepper_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_salt_and_pepper_f=0.3',
  'output_path': 'output_CNMC_4_add_noise_salt_and_pepper_f=0.3'
  }
data['CNMC_4_add_noise_salt_and_pepper_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_salt_and_pepper_f=0.4',
  'output_path': 'output_CNMC_4_add_noise_salt_and_pepper_f=0.4'
  }
data['CNMC_4_add_noise_salt_and_pepper_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_salt_and_pepper_f=0.5',
  'output_path': 'output_CNMC_4_add_noise_salt_and_pepper_f=0.5'
  }
data['CNMC_4_add_noise_salt_and_pepper_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_salt_and_pepper_f=0.6',
  'output_path': 'output_CNMC_4_add_noise_salt_and_pepper_f=0.6'
  }
data['CNMC_4_add_noise_salt_and_pepper_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_salt_and_pepper_f=0.7',
  'output_path': 'output_CNMC_4_add_noise_salt_and_pepper_f=0.7'
  }
data['CNMC_4_add_noise_salt_and_pepper_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_salt_and_pepper_f=0.8',
  'output_path': 'output_CNMC_4_add_noise_salt_and_pepper_f=0.8'
  }
data['CNMC_4_add_noise_salt_and_pepper_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_salt_and_pepper_f=0.9',
  'output_path': 'output_CNMC_4_add_noise_salt_and_pepper_f=0.9'
  }
data['CNMC_4_add_noise_salt_and_pepper_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_salt_and_pepper_f=1.0',
  'output_path': 'output_CNMC_4_add_noise_salt_and_pepper_f=1.0'
  }

data['CNMC_4_add_noise_speckle_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_speckle_f=0.1',
  'output_path': 'output_CNMC_4_add_noise_speckle_f=0.1'
  }
data['CNMC_4_add_noise_speckle_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_speckle_f=0.2',
  'output_path': 'output_CNMC_4_add_noise_speckle_f=0.2'
  }
data['CNMC_4_add_noise_speckle_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_speckle_f=0.3',
  'output_path': 'output_CNMC_4_add_noise_speckle_f=0.3'
  }
data['CNMC_4_add_noise_speckle_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_speckle_f=0.4',
  'output_path': 'output_CNMC_4_add_noise_speckle_f=0.4'
  }
data['CNMC_4_add_noise_speckle_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_speckle_f=0.5',
  'output_path': 'output_CNMC_4_add_noise_speckle_f=0.5'
  }
data['CNMC_4_add_noise_speckle_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_speckle_f=0.6',
  'output_path': 'output_CNMC_4_add_noise_speckle_f=0.6'
  }
data['CNMC_4_add_noise_speckle_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_speckle_f=0.7',
  'output_path': 'output_CNMC_4_add_noise_speckle_f=0.7'
  }
data['CNMC_4_add_noise_speckle_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_speckle_f=0.8',
  'output_path': 'output_CNMC_4_add_noise_speckle_f=0.8'
  }
data['CNMC_4_add_noise_speckle_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_speckle_f=0.9',
  'output_path': 'output_CNMC_4_add_noise_speckle_f=0.9'
  }
data['CNMC_4_add_noise_speckle_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_add_noise_speckle_f=1.0',
  'output_path': 'output_CNMC_4_add_noise_speckle_f=1.0'
  }

data['CNMC_4_imbalance_classes_f=0.1'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_imbalance_classes_f=0.1',
'output_path': 'output_CNMC_4_imbalance_classes_f=0.1'
}
data['CNMC_4_imbalance_classes_f=0.2'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_imbalance_classes_f=0.2',
'output_path': 'output_CNMC_4_imbalance_classes_f=0.2'
}
data['CNMC_4_imbalance_classes_f=0.3'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_imbalance_classes_f=0.3',
'output_path': 'output_CNMC_4_imbalance_classes_f=0.3'
}
data['CNMC_4_imbalance_classes_f=0.4'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_imbalance_classes_f=0.4',
'output_path': 'output_CNMC_4_imbalance_classes_f=0.4'
}
data['CNMC_4_imbalance_classes_f=0.5'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_imbalance_classes_f=0.5',
'output_path': 'output_CNMC_4_imbalance_classes_f=0.5'
}
data['CNMC_4_imbalance_classes_f=0.6'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_imbalance_classes_f=0.6',
'output_path': 'output_CNMC_4_imbalance_classes_f=0.6'
}
data['CNMC_4_imbalance_classes_f=0.7'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_imbalance_classes_f=0.7',
'output_path': 'output_CNMC_4_imbalance_classes_f=0.7'
}
data['CNMC_4_imbalance_classes_f=0.8'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_imbalance_classes_f=0.8',
'output_path': 'output_CNMC_4_imbalance_classes_f=0.8'
}
data['CNMC_4_imbalance_classes_f=0.9'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_imbalance_classes_f=0.9',
'output_path': 'output_CNMC_4_imbalance_classes_f=0.9'
}
data['CNMC_4_imbalance_classes_f=1.0'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_imbalance_classes_f=1.0',
'output_path': 'output_CNMC_4_imbalance_classes_f=1.0'
}

data['CNMC_4_grayscale_f=0.1'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_grayscale_f=0.1',
'output_path': 'output_CNMC_4_grayscale_f=0.1'
}
data['CNMC_4_grayscale_f=0.2'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_grayscale_f=0.2',
'output_path': 'output_CNMC_4_grayscale_f=0.2'
}
data['CNMC_4_grayscale_f=0.3'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_grayscale_f=0.3',
'output_path': 'output_CNMC_4_grayscale_f=0.3'
}
data['CNMC_4_grayscale_f=0.4'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_grayscale_f=0.4',
'output_path': 'output_CNMC_4_grayscale_f=0.4'
}
data['CNMC_4_grayscale_f=0.5'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_grayscale_f=0.5',
'output_path': 'output_CNMC_4_grayscale_f=0.5'
}
data['CNMC_4_grayscale_f=0.6'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_grayscale_f=0.6',
'output_path': 'output_CNMC_4_grayscale_f=0.6'
}
data['CNMC_4_grayscale_f=0.7'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_grayscale_f=0.7',
'output_path': 'output_CNMC_4_grayscale_f=0.7'
}
data['CNMC_4_grayscale_f=0.8'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_grayscale_f=0.8',
'output_path': 'output_CNMC_4_grayscale_f=0.8'
}
data['CNMC_4_grayscale_f=0.9'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_grayscale_f=0.9',
'output_path': 'output_CNMC_4_grayscale_f=0.9'
}
data['CNMC_4_grayscale_f=1.0'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_grayscale_f=1.0',
'output_path': 'output_CNMC_4_grayscale_f=1.0'
}
data['CNMC_4_hsv_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_hsv_f=0.1',
'output_path': 'output_CNMC_4_hsv_f=0.1'
}
data['CNMC_4_hsv_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_hsv_f=0.2',
'output_path': 'output_CNMC_4_hsv_f=0.2'
}
data['CNMC_4_hsv_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_hsv_f=0.3',
'output_path': 'output_CNMC_4_hsv_f=0.3'
}
data['CNMC_4_hsv_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_hsv_f=0.4',
'output_path': 'output_CNMC_4_hsv_f=0.4'
}
data['CNMC_4_hsv_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_hsv_f=0.5',
'output_path': 'output_CNMC_4_hsv_f=0.5'
}
data['CNMC_4_hsv_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_hsv_f=0.6',
'output_path': 'output_CNMC_4_hsv_f=0.6'
}
data['CNMC_4_hsv_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_hsv_f=0.7',
'output_path': 'output_CNMC_4_hsv_f=0.7'
}
data['CNMC_4_hsv_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_hsv_f=0.8',
'output_path': 'output_CNMC_4_hsv_f=0.8'
}
data['CNMC_4_hsv_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_hsv_f=0.9',
'output_path': 'output_CNMC_4_hsv_f=0.9'
}
data['CNMC_4_hsv_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_hsv_f=1.0',
'output_path': 'output_CNMC_4_hsv_f=1.0'
}
data['CNMC_4_blur_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_blur_f=0.1',
'output_path': 'output_CNMC_4_blur_f=0.1'
}
data['CNMC_4_blur_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_blur_f=0.2',
'output_path': 'output_CNMC_4_blur_f=0.2'
}
data['CNMC_4_blur_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_blur_f=0.3',
'output_path': 'output_CNMC_4_blur_f=0.3'
}
data['CNMC_4_blur_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_blur_f=0.4',
'output_path': 'output_CNMC_4_blur_f=0.4'
}
data['CNMC_4_blur_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_blur_f=0.5',
'output_path': 'output_CNMC_4_blur_f=0.5'
}
data['CNMC_4_blur_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_blur_f=0.6',
'output_path': 'output_CNMC_4_blur_f=0.6'
}
data['CNMC_4_blur_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_blur_f=0.7',
'output_path': 'output_CNMC_4_blur_f=0.7'
}
data['CNMC_4_blur_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_blur_f=0.8',
'output_path': 'output_CNMC_4_blur_f=0.8'
}
data['CNMC_4_blur_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_blur_f=0.9',
'output_path': 'output_CNMC_4_blur_f=0.9'
}
data['CNMC_4_blur_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_blur_f=1.0',
'output_path': 'output_CNMC_4_blur_f=1.0'
}

data['CNMC_4_small_random_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_small_random_f=0.1',
'output_path': 'output_CNMC_4_small_random_f=0.1'
}
data['CNMC_4_small_random_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_small_random_f=0.2',
'output_path': 'output_CNMC_4_small_random_f=0.2'
}
data['CNMC_4_small_random_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_small_random_f=0.3',
'output_path': 'output_CNMC_4_small_random_f=0.3'
}
data['CNMC_4_small_random_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_small_random_f=0.4',
'output_path': 'output_CNMC_4_small_random_f=0.4'
}
data['CNMC_4_small_random_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_small_random_f=0.5',
'output_path': 'output_CNMC_4_small_random_f=0.5'
}
data['CNMC_4_small_random_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_small_random_f=0.6',
'output_path': 'output_CNMC_4_small_random_f=0.6'
}
data['CNMC_4_small_random_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_small_random_f=0.7',
'output_path': 'output_CNMC_4_small_random_f=0.7'
}
data['CNMC_4_small_random_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_small_random_f=0.8',
'output_path': 'output_CNMC_4_small_random_f=0.8'
}
data['CNMC_4_small_random_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_small_random_f=0.9',
'output_path': 'output_CNMC_4_small_random_f=0.9'
}
data['CNMC_4_small_random_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_small_random_f=1.0',
'output_path': 'output_CNMC_4_small_random_f=1.0'
}

data['CNMC_4_small_easy_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_small_easy_f=0.1',
  'output_path': 'output_CNMC_4_small_easy_f=0.1'
  }
data['CNMC_4_small_easy_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_small_easy_f=0.2',
  'output_path': 'output_CNMC_4_small_easy_f=0.2'
  }
data['CNMC_4_small_easy_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_small_easy_f=0.3',
  'output_path': 'output_CNMC_4_small_easy_f=0.3'
  }
data['CNMC_4_small_easy_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_small_easy_f=0.4',
  'output_path': 'output_CNMC_4_small_easy_f=0.4'
  }
data['CNMC_4_small_easy_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_small_easy_f=0.5',
  'output_path': 'output_CNMC_4_small_easy_f=0.5'
  }
data['CNMC_4_small_easy_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_small_easy_f=0.6',
  'output_path': 'output_CNMC_4_small_easy_f=0.6'
  }
data['CNMC_4_small_easy_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_small_easy_f=0.7',
  'output_path': 'output_CNMC_4_small_easy_f=0.7'
  }
data['CNMC_4_small_easy_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_small_easy_f=0.8',
  'output_path': 'output_CNMC_4_small_easy_f=0.8'
  }
data['CNMC_4_small_easy_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_small_easy_f=0.9',
  'output_path': 'output_CNMC_4_small_easy_f=0.9'
  }
data['CNMC_4_small_easy_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_small_easy_f=1.0',
  'output_path': 'output_CNMC_4_small_easy_f=1.0'
  }

data['CNMC_4_small_hard_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_small_hard_f=0.1',
  'output_path': 'output_CNMC_4_small_hard_f=0.1'
  }
data['CNMC_4_small_hard_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_small_hard_f=0.2',
  'output_path': 'output_CNMC_4_small_hard_f=0.2'
  }
data['CNMC_4_small_hard_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_small_hard_f=0.3',
  'output_path': 'output_CNMC_4_small_hard_f=0.3'
  }
data['CNMC_4_small_hard_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_small_hard_f=0.4',
  'output_path': 'output_CNMC_4_small_hard_f=0.4'
  }
data['CNMC_4_small_hard_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_small_hard_f=0.5',
  'output_path': 'output_CNMC_4_small_hard_f=0.5'
  }
data['CNMC_4_small_hard_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_small_hard_f=0.6',
  'output_path': 'output_CNMC_4_small_hard_f=0.6'
  }
data['CNMC_4_small_hard_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_small_hard_f=0.7',
  'output_path': 'output_CNMC_4_small_hard_f=0.7'
  }
data['CNMC_4_small_hard_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_small_hard_f=0.8',
  'output_path': 'output_CNMC_4_small_hard_f=0.8'
  }
data['CNMC_4_small_hard_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_small_hard_f=0.9',
  'output_path': 'output_CNMC_4_small_hard_f=0.9'
  }
data['CNMC_4_small_hard_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_4',
  'dataset_path': 'dataset_CNMC_4_small_hard_f=1.0',
  'output_path': 'output_CNMC_4_small_hard_f=1.0'
  }

data['CNMC_4_small_clusters_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_small_clusters_f=0.1',
'output_path': 'output_CNMC_4_small_clusters_f=0.1'
}
data['CNMC_4_small_clusters_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_small_clusters_f=0.2',
'output_path': 'output_CNMC_4_small_clusters_f=0.2'
}
data['CNMC_4_small_clusters_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_small_clusters_f=0.3',
'output_path': 'output_CNMC_4_small_clusters_f=0.3'
}
data['CNMC_4_small_clusters_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_small_clusters_f=0.4',
'output_path': 'output_CNMC_4_small_clusters_f=0.4'
}
data['CNMC_4_small_clusters_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_small_clusters_f=0.5',
'output_path': 'output_CNMC_4_small_clusters_f=0.5'
}
data['CNMC_4_small_clusters_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_small_clusters_f=0.6',
'output_path': 'output_CNMC_4_small_clusters_f=0.6'
}
data['CNMC_4_small_clusters_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_small_clusters_f=0.7',
'output_path': 'output_CNMC_4_small_clusters_f=0.7'
}
data['CNMC_4_small_clusters_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_small_clusters_f=0.8',
'output_path': 'output_CNMC_4_small_clusters_f=0.8'
}
data['CNMC_4_small_clusters_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_small_clusters_f=0.9',
'output_path': 'output_CNMC_4_small_clusters_f=0.9'
}
data['CNMC_4_small_clusters_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_4',
'dataset_path': 'dataset_CNMC_4_small_clusters_f=1.0',
'output_path': 'output_CNMC_4_small_clusters_f=1.0'
}

data['CNMC_5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5',
  'output_path': 'output_CNMC_5'
  }
data['CNMC_5_image_rot_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_image_rot_f=0.1',
  'output_path': 'output_CNMC_5_image_rot_f=0.1'
  }
data['CNMC_5_image_rot_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_image_rot_f=0.2',
  'output_path': 'output_CNMC_5_image_rot_f=0.2'
  }
data['CNMC_5_image_rot_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_image_rot_f=0.3',
  'output_path': 'output_CNMC_5_image_rot_f=0.3'
  }
data['CNMC_5_image_rot_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_image_rot_f=0.4',
  'output_path': 'output_CNMC_5_image_rot_f=0.4'
  }
data['CNMC_5_image_rot_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_image_rot_f=0.5',
  'output_path': 'output_CNMC_5_image_rot_f=0.5'
  }
data['CNMC_5_image_rot_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_image_rot_f=0.6',
  'output_path': 'output_CNMC_5_image_rot_f=0.6'
  }
data['CNMC_5_image_rot_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_image_rot_f=0.7',
  'output_path': 'output_CNMC_5_image_rot_f=0.7'
  }
data['CNMC_5_image_rot_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_image_rot_f=0.8',
  'output_path': 'output_CNMC_5_image_rot_f=0.8'
  }
data['CNMC_5_image_rot_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_image_rot_f=0.9',
  'output_path': 'output_CNMC_5_image_rot_f=0.9'
  }
data['CNMC_5_image_rot_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_image_rot_f=1.0',
  'output_path': 'output_CNMC_5_image_rot_f=1.0'
  }

data['CNMC_5_image_translation_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_image_translation_f=0.1',
  'output_path': 'output_CNMC_5_image_translation_f=0.1'
  }
data['CNMC_5_image_translation_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_image_translation_f=0.2',
  'output_path': 'output_CNMC_5_image_translation_f=0.2'
  }
data['CNMC_5_image_translation_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_image_translation_f=0.3',
  'output_path': 'output_CNMC_5_image_translation_f=0.3'
  }
data['CNMC_5_image_translation_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_image_translation_f=0.4',
  'output_path': 'output_CNMC_5_image_translation_f=0.4'
  }
data['CNMC_5_image_translation_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_image_translation_f=0.5',
  'output_path': 'output_CNMC_5_image_translation_f=0.5'
  }
data['CNMC_5_image_translation_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_image_translation_f=0.6',
  'output_path': 'output_CNMC_5_image_translation_f=0.6'
  }
data['CNMC_5_image_translation_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_image_translation_f=0.7',
  'output_path': 'output_CNMC_5_image_translation_f=0.7'
  }
data['CNMC_5_image_translation_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_image_translation_f=0.8',
  'output_path': 'output_CNMC_5_image_translation_f=0.8'
  }
data['CNMC_5_image_translation_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_image_translation_f=0.9',
  'output_path': 'output_CNMC_5_image_translation_f=0.9'
  }
data['CNMC_5_image_translation_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_image_translation_f=1.0',
  'output_path': 'output_CNMC_5_image_translation_f=1.0'
  }

data['CNMC_5_image_zoom_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_image_zoom_f=0.1',
  'output_path': 'output_CNMC_5_image_zoom_f=0.1'
  }
data['CNMC_5_image_zoom_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_image_zoom_f=0.2',
  'output_path': 'output_CNMC_5_image_zoom_f=0.2'
  }
data['CNMC_5_image_zoom_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_image_zoom_f=0.3',
  'output_path': 'output_CNMC_5_image_zoom_f=0.3'
  }
data['CNMC_5_image_zoom_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_image_zoom_f=0.4',
  'output_path': 'output_CNMC_5_image_zoom_f=0.4'
  }
data['CNMC_5_image_zoom_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_image_zoom_f=0.5',
  'output_path': 'output_CNMC_5_image_zoom_f=0.5'
  }
data['CNMC_5_image_zoom_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_image_zoom_f=0.6',
  'output_path': 'output_CNMC_5_image_zoom_f=0.6'
  }
data['CNMC_5_image_zoom_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_image_zoom_f=0.7',
  'output_path': 'output_CNMC_5_image_zoom_f=0.7'
  }
data['CNMC_5_image_zoom_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_image_zoom_f=0.8',
  'output_path': 'output_CNMC_5_image_zoom_f=0.8'
  }
data['CNMC_5_image_zoom_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_image_zoom_f=0.9',
  'output_path': 'output_CNMC_5_image_zoom_f=0.9'
  }
data['CNMC_5_image_zoom_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_image_zoom_f=1.0',
  'output_path': 'output_CNMC_5_image_zoom_f=1.0'
  }

data['CNMC_5_add_noise_gaussian_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_gaussian_f=0.1',
  'output_path': 'output_CNMC_5_add_noise_gaussian_f=0.1'
  }
data['CNMC_5_add_noise_gaussian_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_gaussian_f=0.2',
  'output_path': 'output_CNMC_5_add_noise_gaussian_f=0.2'
  }
data['CNMC_5_add_noise_gaussian_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_gaussian_f=0.3',
  'output_path': 'output_CNMC_5_add_noise_gaussian_f=0.3'
  }
data['CNMC_5_add_noise_gaussian_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_gaussian_f=0.4',
  'output_path': 'output_CNMC_5_add_noise_gaussian_f=0.4'
  }
data['CNMC_5_add_noise_gaussian_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_gaussian_f=0.5',
  'output_path': 'output_CNMC_5_add_noise_gaussian_f=0.5'
  }
data['CNMC_5_add_noise_gaussian_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_gaussian_f=0.6',
  'output_path': 'output_CNMC_5_add_noise_gaussian_f=0.6'
  }
data['CNMC_5_add_noise_gaussian_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_gaussian_f=0.7',
  'output_path': 'output_CNMC_5_add_noise_gaussian_f=0.7'
  }
data['CNMC_5_add_noise_gaussian_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_gaussian_f=0.8',
  'output_path': 'output_CNMC_5_add_noise_gaussian_f=0.8'
  }
data['CNMC_5_add_noise_gaussian_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_gaussian_f=0.9',
  'output_path': 'output_CNMC_5_add_noise_gaussian_f=0.9'
  }
data['CNMC_5_add_noise_gaussian_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_gaussian_f=1.0',
  'output_path': 'output_CNMC_5_add_noise_gaussian_f=1.0'
  }

data['CNMC_5_add_noise_poisson_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_poisson_f=0.1',
  'output_path': 'output_CNMC_5_add_noise_poisson_f=0.1'
  }
data['CNMC_5_add_noise_poisson_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_poisson_f=0.2',
  'output_path': 'output_CNMC_5_add_noise_poisson_f=0.2'
  }
data['CNMC_5_add_noise_poisson_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_poisson_f=0.3',
  'output_path': 'output_CNMC_5_add_noise_poisson_f=0.3'
  }
data['CNMC_5_add_noise_poisson_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_poisson_f=0.4',
  'output_path': 'output_CNMC_5_add_noise_poisson_f=0.4'
  }
data['CNMC_5_add_noise_poisson_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_poisson_f=0.5',
  'output_path': 'output_CNMC_5_add_noise_poisson_f=0.5'
  }
data['CNMC_5_add_noise_poisson_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_poisson_f=0.6',
  'output_path': 'output_CNMC_5_add_noise_poisson_f=0.6'
  }
data['CNMC_5_add_noise_poisson_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_poisson_f=0.7',
  'output_path': 'output_CNMC_5_add_noise_poisson_f=0.7'
  }
data['CNMC_5_add_noise_poisson_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_poisson_f=0.8',
  'output_path': 'output_CNMC_5_add_noise_poisson_f=0.8'
  }
data['CNMC_5_add_noise_poisson_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_poisson_f=0.9',
  'output_path': 'output_CNMC_5_add_noise_poisson_f=0.9'
  }
data['CNMC_5_add_noise_poisson_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_poisson_f=1.0',
  'output_path': 'output_CNMC_5_add_noise_poisson_f=1.0'
  }

data['CNMC_5_add_noise_salt_and_pepper_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_salt_and_pepper_f=0.1',
  'output_path': 'output_CNMC_5_add_noise_salt_and_pepper_f=0.1'
  }
data['CNMC_5_add_noise_salt_and_pepper_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_salt_and_pepper_f=0.2',
  'output_path': 'output_CNMC_5_add_noise_salt_and_pepper_f=0.2'
  }
data['CNMC_5_add_noise_salt_and_pepper_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_salt_and_pepper_f=0.3',
  'output_path': 'output_CNMC_5_add_noise_salt_and_pepper_f=0.3'
  }
data['CNMC_5_add_noise_salt_and_pepper_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_salt_and_pepper_f=0.4',
  'output_path': 'output_CNMC_5_add_noise_salt_and_pepper_f=0.4'
  }
data['CNMC_5_add_noise_salt_and_pepper_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_salt_and_pepper_f=0.5',
  'output_path': 'output_CNMC_5_add_noise_salt_and_pepper_f=0.5'
  }
data['CNMC_5_add_noise_salt_and_pepper_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_salt_and_pepper_f=0.6',
  'output_path': 'output_CNMC_5_add_noise_salt_and_pepper_f=0.6'
  }
data['CNMC_5_add_noise_salt_and_pepper_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_salt_and_pepper_f=0.7',
  'output_path': 'output_CNMC_5_add_noise_salt_and_pepper_f=0.7'
  }
data['CNMC_5_add_noise_salt_and_pepper_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_salt_and_pepper_f=0.8',
  'output_path': 'output_CNMC_5_add_noise_salt_and_pepper_f=0.8'
  }
data['CNMC_5_add_noise_salt_and_pepper_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_salt_and_pepper_f=0.9',
  'output_path': 'output_CNMC_5_add_noise_salt_and_pepper_f=0.9'
  }
data['CNMC_5_add_noise_salt_and_pepper_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_salt_and_pepper_f=1.0',
  'output_path': 'output_CNMC_5_add_noise_salt_and_pepper_f=1.0'
  }

data['CNMC_5_add_noise_speckle_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_speckle_f=0.1',
  'output_path': 'output_CNMC_5_add_noise_speckle_f=0.1'
  }
data['CNMC_5_add_noise_speckle_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_speckle_f=0.2',
  'output_path': 'output_CNMC_5_add_noise_speckle_f=0.2'
  }
data['CNMC_5_add_noise_speckle_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_speckle_f=0.3',
  'output_path': 'output_CNMC_5_add_noise_speckle_f=0.3'
  }
data['CNMC_5_add_noise_speckle_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_speckle_f=0.4',
  'output_path': 'output_CNMC_5_add_noise_speckle_f=0.4'
  }
data['CNMC_5_add_noise_speckle_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_speckle_f=0.5',
  'output_path': 'output_CNMC_5_add_noise_speckle_f=0.5'
  }
data['CNMC_5_add_noise_speckle_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_speckle_f=0.6',
  'output_path': 'output_CNMC_5_add_noise_speckle_f=0.6'
  }
data['CNMC_5_add_noise_speckle_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_speckle_f=0.7',
  'output_path': 'output_CNMC_5_add_noise_speckle_f=0.7'
  }
data['CNMC_5_add_noise_speckle_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_speckle_f=0.8',
  'output_path': 'output_CNMC_5_add_noise_speckle_f=0.8'
  }
data['CNMC_5_add_noise_speckle_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_speckle_f=0.9',
  'output_path': 'output_CNMC_5_add_noise_speckle_f=0.9'
  }
data['CNMC_5_add_noise_speckle_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_add_noise_speckle_f=1.0',
  'output_path': 'output_CNMC_5_add_noise_speckle_f=1.0'
  }

data['CNMC_5_imbalance_classes_f=0.1'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_imbalance_classes_f=0.1',
'output_path': 'output_CNMC_5_imbalance_classes_f=0.1'
}
data['CNMC_5_imbalance_classes_f=0.2'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_imbalance_classes_f=0.2',
'output_path': 'output_CNMC_5_imbalance_classes_f=0.2'
}
data['CNMC_5_imbalance_classes_f=0.3'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_imbalance_classes_f=0.3',
'output_path': 'output_CNMC_5_imbalance_classes_f=0.3'
}
data['CNMC_5_imbalance_classes_f=0.4'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_imbalance_classes_f=0.4',
'output_path': 'output_CNMC_5_imbalance_classes_f=0.4'
}
data['CNMC_5_imbalance_classes_f=0.5'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_imbalance_classes_f=0.5',
'output_path': 'output_CNMC_5_imbalance_classes_f=0.5'
}
data['CNMC_5_imbalance_classes_f=0.6'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_imbalance_classes_f=0.6',
'output_path': 'output_CNMC_5_imbalance_classes_f=0.6'
}
data['CNMC_5_imbalance_classes_f=0.7'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_imbalance_classes_f=0.7',
'output_path': 'output_CNMC_5_imbalance_classes_f=0.7'
}
data['CNMC_5_imbalance_classes_f=0.8'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_imbalance_classes_f=0.8',
'output_path': 'output_CNMC_5_imbalance_classes_f=0.8'
}
data['CNMC_5_imbalance_classes_f=0.9'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_imbalance_classes_f=0.9',
'output_path': 'output_CNMC_5_imbalance_classes_f=0.9'
}
data['CNMC_5_imbalance_classes_f=1.0'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_imbalance_classes_f=1.0',
'output_path': 'output_CNMC_5_imbalance_classes_f=1.0'
}

data['CNMC_5_grayscale_f=0.1'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_grayscale_f=0.1',
'output_path': 'output_CNMC_5_grayscale_f=0.1'
}
data['CNMC_5_grayscale_f=0.2'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_grayscale_f=0.2',
'output_path': 'output_CNMC_5_grayscale_f=0.2'
}
data['CNMC_5_grayscale_f=0.3'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_grayscale_f=0.3',
'output_path': 'output_CNMC_5_grayscale_f=0.3'
}
data['CNMC_5_grayscale_f=0.4'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_grayscale_f=0.4',
'output_path': 'output_CNMC_5_grayscale_f=0.4'
}
data['CNMC_5_grayscale_f=0.5'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_grayscale_f=0.5',
'output_path': 'output_CNMC_5_grayscale_f=0.5'
}
data['CNMC_5_grayscale_f=0.6'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_grayscale_f=0.6',
'output_path': 'output_CNMC_5_grayscale_f=0.6'
}
data['CNMC_5_grayscale_f=0.7'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_grayscale_f=0.7',
'output_path': 'output_CNMC_5_grayscale_f=0.7'
}
data['CNMC_5_grayscale_f=0.8'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_grayscale_f=0.8',
'output_path': 'output_CNMC_5_grayscale_f=0.8'
}
data['CNMC_5_grayscale_f=0.9'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_grayscale_f=0.9',
'output_path': 'output_CNMC_5_grayscale_f=0.9'
}
data['CNMC_5_grayscale_f=1.0'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_grayscale_f=1.0',
'output_path': 'output_CNMC_5_grayscale_f=1.0'
}
data['CNMC_5_hsv_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_hsv_f=0.1',
'output_path': 'output_CNMC_5_hsv_f=0.1'
}
data['CNMC_5_hsv_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_hsv_f=0.2',
'output_path': 'output_CNMC_5_hsv_f=0.2'
}
data['CNMC_5_hsv_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_hsv_f=0.3',
'output_path': 'output_CNMC_5_hsv_f=0.3'
}
data['CNMC_5_hsv_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_hsv_f=0.4',
'output_path': 'output_CNMC_5_hsv_f=0.4'
}
data['CNMC_5_hsv_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_hsv_f=0.5',
'output_path': 'output_CNMC_5_hsv_f=0.5'
}
data['CNMC_5_hsv_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_hsv_f=0.6',
'output_path': 'output_CNMC_5_hsv_f=0.6'
}
data['CNMC_5_hsv_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_hsv_f=0.7',
'output_path': 'output_CNMC_5_hsv_f=0.7'
}
data['CNMC_5_hsv_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_hsv_f=0.8',
'output_path': 'output_CNMC_5_hsv_f=0.8'
}
data['CNMC_5_hsv_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_hsv_f=0.9',
'output_path': 'output_CNMC_5_hsv_f=0.9'
}
data['CNMC_5_hsv_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_hsv_f=1.0',
'output_path': 'output_CNMC_5_hsv_f=1.0'
}
data['CNMC_5_blur_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_blur_f=0.1',
'output_path': 'output_CNMC_5_blur_f=0.1'
}
data['CNMC_5_blur_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_blur_f=0.2',
'output_path': 'output_CNMC_5_blur_f=0.2'
}
data['CNMC_5_blur_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_blur_f=0.3',
'output_path': 'output_CNMC_5_blur_f=0.3'
}
data['CNMC_5_blur_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_blur_f=0.4',
'output_path': 'output_CNMC_5_blur_f=0.4'
}
data['CNMC_5_blur_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_blur_f=0.5',
'output_path': 'output_CNMC_5_blur_f=0.5'
}
data['CNMC_5_blur_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_blur_f=0.6',
'output_path': 'output_CNMC_5_blur_f=0.6'
}
data['CNMC_5_blur_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_blur_f=0.7',
'output_path': 'output_CNMC_5_blur_f=0.7'
}
data['CNMC_5_blur_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_blur_f=0.8',
'output_path': 'output_CNMC_5_blur_f=0.8'
}
data['CNMC_5_blur_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_blur_f=0.9',
'output_path': 'output_CNMC_5_blur_f=0.9'
}
data['CNMC_5_blur_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_blur_f=1.0',
'output_path': 'output_CNMC_5_blur_f=1.0'
}

data['CNMC_5_small_random_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_small_random_f=0.1',
'output_path': 'output_CNMC_5_small_random_f=0.1'
}
data['CNMC_5_small_random_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_small_random_f=0.2',
'output_path': 'output_CNMC_5_small_random_f=0.2'
}
data['CNMC_5_small_random_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_small_random_f=0.3',
'output_path': 'output_CNMC_5_small_random_f=0.3'
}
data['CNMC_5_small_random_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_small_random_f=0.4',
'output_path': 'output_CNMC_5_small_random_f=0.4'
}
data['CNMC_5_small_random_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_small_random_f=0.5',
'output_path': 'output_CNMC_5_small_random_f=0.5'
}
data['CNMC_5_small_random_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_small_random_f=0.6',
'output_path': 'output_CNMC_5_small_random_f=0.6'
}
data['CNMC_5_small_random_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_small_random_f=0.7',
'output_path': 'output_CNMC_5_small_random_f=0.7'
}
data['CNMC_5_small_random_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_small_random_f=0.8',
'output_path': 'output_CNMC_5_small_random_f=0.8'
}
data['CNMC_5_small_random_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_small_random_f=0.9',
'output_path': 'output_CNMC_5_small_random_f=0.9'
}
data['CNMC_5_small_random_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_small_random_f=1.0',
'output_path': 'output_CNMC_5_small_random_f=1.0'
}

data['CNMC_5_small_easy_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_small_easy_f=0.1',
  'output_path': 'output_CNMC_5_small_easy_f=0.1'
  }
data['CNMC_5_small_easy_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_small_easy_f=0.2',
  'output_path': 'output_CNMC_5_small_easy_f=0.2'
  }
data['CNMC_5_small_easy_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_small_easy_f=0.3',
  'output_path': 'output_CNMC_5_small_easy_f=0.3'
  }
data['CNMC_5_small_easy_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_small_easy_f=0.4',
  'output_path': 'output_CNMC_5_small_easy_f=0.4'
  }
data['CNMC_5_small_easy_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_small_easy_f=0.5',
  'output_path': 'output_CNMC_5_small_easy_f=0.5'
  }
data['CNMC_5_small_easy_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_small_easy_f=0.6',
  'output_path': 'output_CNMC_5_small_easy_f=0.6'
  }
data['CNMC_5_small_easy_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_small_easy_f=0.7',
  'output_path': 'output_CNMC_5_small_easy_f=0.7'
  }
data['CNMC_5_small_easy_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_small_easy_f=0.8',
  'output_path': 'output_CNMC_5_small_easy_f=0.8'
  }
data['CNMC_5_small_easy_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_small_easy_f=0.9',
  'output_path': 'output_CNMC_5_small_easy_f=0.9'
  }
data['CNMC_5_small_easy_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_small_easy_f=1.0',
  'output_path': 'output_CNMC_5_small_easy_f=1.0'
  }

data['CNMC_5_small_hard_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_small_hard_f=0.1',
  'output_path': 'output_CNMC_5_small_hard_f=0.1'
  }
data['CNMC_5_small_hard_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_small_hard_f=0.2',
  'output_path': 'output_CNMC_5_small_hard_f=0.2'
  }
data['CNMC_5_small_hard_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_small_hard_f=0.3',
  'output_path': 'output_CNMC_5_small_hard_f=0.3'
  }
data['CNMC_5_small_hard_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_small_hard_f=0.4',
  'output_path': 'output_CNMC_5_small_hard_f=0.4'
  }
data['CNMC_5_small_hard_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_small_hard_f=0.5',
  'output_path': 'output_CNMC_5_small_hard_f=0.5'
  }
data['CNMC_5_small_hard_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_small_hard_f=0.6',
  'output_path': 'output_CNMC_5_small_hard_f=0.6'
  }
data['CNMC_5_small_hard_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_small_hard_f=0.7',
  'output_path': 'output_CNMC_5_small_hard_f=0.7'
  }
data['CNMC_5_small_hard_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_small_hard_f=0.8',
  'output_path': 'output_CNMC_5_small_hard_f=0.8'
  }
data['CNMC_5_small_hard_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_small_hard_f=0.9',
  'output_path': 'output_CNMC_5_small_hard_f=0.9'
  }
data['CNMC_5_small_hard_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_5',
  'dataset_path': 'dataset_CNMC_5_small_hard_f=1.0',
  'output_path': 'output_CNMC_5_small_hard_f=1.0'
  }

data['CNMC_5_small_clusters_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_small_clusters_f=0.1',
'output_path': 'output_CNMC_5_small_clusters_f=0.1'
}
data['CNMC_5_small_clusters_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_small_clusters_f=0.2',
'output_path': 'output_CNMC_5_small_clusters_f=0.2'
}
data['CNMC_5_small_clusters_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_small_clusters_f=0.3',
'output_path': 'output_CNMC_5_small_clusters_f=0.3'
}
data['CNMC_5_small_clusters_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_small_clusters_f=0.4',
'output_path': 'output_CNMC_5_small_clusters_f=0.4'
}
data['CNMC_5_small_clusters_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_small_clusters_f=0.5',
'output_path': 'output_CNMC_5_small_clusters_f=0.5'
}
data['CNMC_5_small_clusters_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_small_clusters_f=0.6',
'output_path': 'output_CNMC_5_small_clusters_f=0.6'
}
data['CNMC_5_small_clusters_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_small_clusters_f=0.7',
'output_path': 'output_CNMC_5_small_clusters_f=0.7'
}
data['CNMC_5_small_clusters_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_small_clusters_f=0.8',
'output_path': 'output_CNMC_5_small_clusters_f=0.8'
}
data['CNMC_5_small_clusters_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_small_clusters_f=0.9',
'output_path': 'output_CNMC_5_small_clusters_f=0.9'
}
data['CNMC_5_small_clusters_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_5',
'dataset_path': 'dataset_CNMC_5_small_clusters_f=1.0',
'output_path': 'output_CNMC_5_small_clusters_f=1.0'
}

data['CNMC_6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6',
  'output_path': 'output_CNMC_6'
  }
data['CNMC_6_image_rot_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_image_rot_f=0.1',
  'output_path': 'output_CNMC_6_image_rot_f=0.1'
  }
data['CNMC_6_image_rot_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_image_rot_f=0.2',
  'output_path': 'output_CNMC_6_image_rot_f=0.2'
  }
data['CNMC_6_image_rot_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_image_rot_f=0.3',
  'output_path': 'output_CNMC_6_image_rot_f=0.3'
  }
data['CNMC_6_image_rot_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_image_rot_f=0.4',
  'output_path': 'output_CNMC_6_image_rot_f=0.4'
  }
data['CNMC_6_image_rot_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_image_rot_f=0.5',
  'output_path': 'output_CNMC_6_image_rot_f=0.5'
  }
data['CNMC_6_image_rot_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_image_rot_f=0.6',
  'output_path': 'output_CNMC_6_image_rot_f=0.6'
  }
data['CNMC_6_image_rot_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_image_rot_f=0.7',
  'output_path': 'output_CNMC_6_image_rot_f=0.7'
  }
data['CNMC_6_image_rot_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_image_rot_f=0.8',
  'output_path': 'output_CNMC_6_image_rot_f=0.8'
  }
data['CNMC_6_image_rot_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_image_rot_f=0.9',
  'output_path': 'output_CNMC_6_image_rot_f=0.9'
  }
data['CNMC_6_image_rot_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_image_rot_f=1.0',
  'output_path': 'output_CNMC_6_image_rot_f=1.0'
  }

data['CNMC_6_image_translation_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_image_translation_f=0.1',
  'output_path': 'output_CNMC_6_image_translation_f=0.1'
  }
data['CNMC_6_image_translation_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_image_translation_f=0.2',
  'output_path': 'output_CNMC_6_image_translation_f=0.2'
  }
data['CNMC_6_image_translation_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_image_translation_f=0.3',
  'output_path': 'output_CNMC_6_image_translation_f=0.3'
  }
data['CNMC_6_image_translation_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_image_translation_f=0.4',
  'output_path': 'output_CNMC_6_image_translation_f=0.4'
  }
data['CNMC_6_image_translation_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_image_translation_f=0.5',
  'output_path': 'output_CNMC_6_image_translation_f=0.5'
  }
data['CNMC_6_image_translation_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_image_translation_f=0.6',
  'output_path': 'output_CNMC_6_image_translation_f=0.6'
  }
data['CNMC_6_image_translation_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_image_translation_f=0.7',
  'output_path': 'output_CNMC_6_image_translation_f=0.7'
  }
data['CNMC_6_image_translation_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_image_translation_f=0.8',
  'output_path': 'output_CNMC_6_image_translation_f=0.8'
  }
data['CNMC_6_image_translation_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_image_translation_f=0.9',
  'output_path': 'output_CNMC_6_image_translation_f=0.9'
  }
data['CNMC_6_image_translation_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_image_translation_f=1.0',
  'output_path': 'output_CNMC_6_image_translation_f=1.0'
  }

data['CNMC_6_image_zoom_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_image_zoom_f=0.1',
  'output_path': 'output_CNMC_6_image_zoom_f=0.1'
  }
data['CNMC_6_image_zoom_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_image_zoom_f=0.2',
  'output_path': 'output_CNMC_6_image_zoom_f=0.2'
  }
data['CNMC_6_image_zoom_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_image_zoom_f=0.3',
  'output_path': 'output_CNMC_6_image_zoom_f=0.3'
  }
data['CNMC_6_image_zoom_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_image_zoom_f=0.4',
  'output_path': 'output_CNMC_6_image_zoom_f=0.4'
  }
data['CNMC_6_image_zoom_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_image_zoom_f=0.5',
  'output_path': 'output_CNMC_6_image_zoom_f=0.5'
  }
data['CNMC_6_image_zoom_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_image_zoom_f=0.6',
  'output_path': 'output_CNMC_6_image_zoom_f=0.6'
  }
data['CNMC_6_image_zoom_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_image_zoom_f=0.7',
  'output_path': 'output_CNMC_6_image_zoom_f=0.7'
  }
data['CNMC_6_image_zoom_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_image_zoom_f=0.8',
  'output_path': 'output_CNMC_6_image_zoom_f=0.8'
  }
data['CNMC_6_image_zoom_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_image_zoom_f=0.9',
  'output_path': 'output_CNMC_6_image_zoom_f=0.9'
  }
data['CNMC_6_image_zoom_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_image_zoom_f=1.0',
  'output_path': 'output_CNMC_6_image_zoom_f=1.0'
  }

data['CNMC_6_add_noise_gaussian_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_gaussian_f=0.1',
  'output_path': 'output_CNMC_6_add_noise_gaussian_f=0.1'
  }
data['CNMC_6_add_noise_gaussian_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_gaussian_f=0.2',
  'output_path': 'output_CNMC_6_add_noise_gaussian_f=0.2'
  }
data['CNMC_6_add_noise_gaussian_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_gaussian_f=0.3',
  'output_path': 'output_CNMC_6_add_noise_gaussian_f=0.3'
  }
data['CNMC_6_add_noise_gaussian_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_gaussian_f=0.4',
  'output_path': 'output_CNMC_6_add_noise_gaussian_f=0.4'
  }
data['CNMC_6_add_noise_gaussian_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_gaussian_f=0.5',
  'output_path': 'output_CNMC_6_add_noise_gaussian_f=0.5'
  }
data['CNMC_6_add_noise_gaussian_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_gaussian_f=0.6',
  'output_path': 'output_CNMC_6_add_noise_gaussian_f=0.6'
  }
data['CNMC_6_add_noise_gaussian_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_gaussian_f=0.7',
  'output_path': 'output_CNMC_6_add_noise_gaussian_f=0.7'
  }
data['CNMC_6_add_noise_gaussian_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_gaussian_f=0.8',
  'output_path': 'output_CNMC_6_add_noise_gaussian_f=0.8'
  }
data['CNMC_6_add_noise_gaussian_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_gaussian_f=0.9',
  'output_path': 'output_CNMC_6_add_noise_gaussian_f=0.9'
  }
data['CNMC_6_add_noise_gaussian_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_gaussian_f=1.0',
  'output_path': 'output_CNMC_6_add_noise_gaussian_f=1.0'
  }

data['CNMC_6_add_noise_poisson_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_poisson_f=0.1',
  'output_path': 'output_CNMC_6_add_noise_poisson_f=0.1'
  }
data['CNMC_6_add_noise_poisson_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_poisson_f=0.2',
  'output_path': 'output_CNMC_6_add_noise_poisson_f=0.2'
  }
data['CNMC_6_add_noise_poisson_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_poisson_f=0.3',
  'output_path': 'output_CNMC_6_add_noise_poisson_f=0.3'
  }
data['CNMC_6_add_noise_poisson_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_poisson_f=0.4',
  'output_path': 'output_CNMC_6_add_noise_poisson_f=0.4'
  }
data['CNMC_6_add_noise_poisson_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_poisson_f=0.5',
  'output_path': 'output_CNMC_6_add_noise_poisson_f=0.5'
  }
data['CNMC_6_add_noise_poisson_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_poisson_f=0.6',
  'output_path': 'output_CNMC_6_add_noise_poisson_f=0.6'
  }
data['CNMC_6_add_noise_poisson_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_poisson_f=0.7',
  'output_path': 'output_CNMC_6_add_noise_poisson_f=0.7'
  }
data['CNMC_6_add_noise_poisson_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_poisson_f=0.8',
  'output_path': 'output_CNMC_6_add_noise_poisson_f=0.8'
  }
data['CNMC_6_add_noise_poisson_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_poisson_f=0.9',
  'output_path': 'output_CNMC_6_add_noise_poisson_f=0.9'
  }
data['CNMC_6_add_noise_poisson_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_poisson_f=1.0',
  'output_path': 'output_CNMC_6_add_noise_poisson_f=1.0'
  }

data['CNMC_6_add_noise_salt_and_pepper_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_salt_and_pepper_f=0.1',
  'output_path': 'output_CNMC_6_add_noise_salt_and_pepper_f=0.1'
  }
data['CNMC_6_add_noise_salt_and_pepper_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_salt_and_pepper_f=0.2',
  'output_path': 'output_CNMC_6_add_noise_salt_and_pepper_f=0.2'
  }
data['CNMC_6_add_noise_salt_and_pepper_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_salt_and_pepper_f=0.3',
  'output_path': 'output_CNMC_6_add_noise_salt_and_pepper_f=0.3'
  }
data['CNMC_6_add_noise_salt_and_pepper_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_salt_and_pepper_f=0.4',
  'output_path': 'output_CNMC_6_add_noise_salt_and_pepper_f=0.4'
  }
data['CNMC_6_add_noise_salt_and_pepper_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_salt_and_pepper_f=0.5',
  'output_path': 'output_CNMC_6_add_noise_salt_and_pepper_f=0.5'
  }
data['CNMC_6_add_noise_salt_and_pepper_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_salt_and_pepper_f=0.6',
  'output_path': 'output_CNMC_6_add_noise_salt_and_pepper_f=0.6'
  }
data['CNMC_6_add_noise_salt_and_pepper_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_salt_and_pepper_f=0.7',
  'output_path': 'output_CNMC_6_add_noise_salt_and_pepper_f=0.7'
  }
data['CNMC_6_add_noise_salt_and_pepper_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_salt_and_pepper_f=0.8',
  'output_path': 'output_CNMC_6_add_noise_salt_and_pepper_f=0.8'
  }
data['CNMC_6_add_noise_salt_and_pepper_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_salt_and_pepper_f=0.9',
  'output_path': 'output_CNMC_6_add_noise_salt_and_pepper_f=0.9'
  }
data['CNMC_6_add_noise_salt_and_pepper_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_salt_and_pepper_f=1.0',
  'output_path': 'output_CNMC_6_add_noise_salt_and_pepper_f=1.0'
  }

data['CNMC_6_add_noise_speckle_f=0.1'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_speckle_f=0.1',
  'output_path': 'output_CNMC_6_add_noise_speckle_f=0.1'
  }
data['CNMC_6_add_noise_speckle_f=0.2'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_speckle_f=0.2',
  'output_path': 'output_CNMC_6_add_noise_speckle_f=0.2'
  }
data['CNMC_6_add_noise_speckle_f=0.3'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_speckle_f=0.3',
  'output_path': 'output_CNMC_6_add_noise_speckle_f=0.3'
  }
data['CNMC_6_add_noise_speckle_f=0.4'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_speckle_f=0.4',
  'output_path': 'output_CNMC_6_add_noise_speckle_f=0.4'
  }
data['CNMC_6_add_noise_speckle_f=0.5'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_speckle_f=0.5',
  'output_path': 'output_CNMC_6_add_noise_speckle_f=0.5'
  }
data['CNMC_6_add_noise_speckle_f=0.6'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_speckle_f=0.6',
  'output_path': 'output_CNMC_6_add_noise_speckle_f=0.6'
  }
data['CNMC_6_add_noise_speckle_f=0.7'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_speckle_f=0.7',
  'output_path': 'output_CNMC_6_add_noise_speckle_f=0.7'
  }
data['CNMC_6_add_noise_speckle_f=0.8'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_speckle_f=0.8',
  'output_path': 'output_CNMC_6_add_noise_speckle_f=0.8'
  }
data['CNMC_6_add_noise_speckle_f=0.9'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_speckle_f=0.9',
  'output_path': 'output_CNMC_6_add_noise_speckle_f=0.9'
  }
data['CNMC_6_add_noise_speckle_f=1.0'] = {
  'classes': ['normal', 'leukemic'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_add_noise_speckle_f=1.0',
  'output_path': 'output_CNMC_6_add_noise_speckle_f=1.0'
  }

data['CNMC_6_imbalance_classes_f=0.1'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_imbalance_classes_f=0.1',
'output_path': 'output_CNMC_6_imbalance_classes_f=0.1'
}
data['CNMC_6_imbalance_classes_f=0.2'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_imbalance_classes_f=0.2',
'output_path': 'output_CNMC_6_imbalance_classes_f=0.2'
}
data['CNMC_6_imbalance_classes_f=0.3'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_imbalance_classes_f=0.3',
'output_path': 'output_CNMC_6_imbalance_classes_f=0.3'
}
data['CNMC_6_imbalance_classes_f=0.4'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_imbalance_classes_f=0.4',
'output_path': 'output_CNMC_6_imbalance_classes_f=0.4'
}
data['CNMC_6_imbalance_classes_f=0.5'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_imbalance_classes_f=0.5',
'output_path': 'output_CNMC_6_imbalance_classes_f=0.5'
}
data['CNMC_6_imbalance_classes_f=0.6'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_imbalance_classes_f=0.6',
'output_path': 'output_CNMC_6_imbalance_classes_f=0.6'
}
data['CNMC_6_imbalance_classes_f=0.7'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_imbalance_classes_f=0.7',
'output_path': 'output_CNMC_6_imbalance_classes_f=0.7'
}
data['CNMC_6_imbalance_classes_f=0.8'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_imbalance_classes_f=0.8',
'output_path': 'output_CNMC_6_imbalance_classes_f=0.8'
}
data['CNMC_6_imbalance_classes_f=0.9'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_imbalance_classes_f=0.9',
'output_path': 'output_CNMC_6_imbalance_classes_f=0.9'
}
data['CNMC_6_imbalance_classes_f=1.0'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_imbalance_classes_f=1.0',
'output_path': 'output_CNMC_6_imbalance_classes_f=1.0'
}

data['CNMC_6_grayscale_f=0.1'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_grayscale_f=0.1',
'output_path': 'output_CNMC_6_grayscale_f=0.1'
}
data['CNMC_6_grayscale_f=0.2'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_grayscale_f=0.2',
'output_path': 'output_CNMC_6_grayscale_f=0.2'
}
data['CNMC_6_grayscale_f=0.3'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_grayscale_f=0.3',
'output_path': 'output_CNMC_6_grayscale_f=0.3'
}
data['CNMC_6_grayscale_f=0.4'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_grayscale_f=0.4',
'output_path': 'output_CNMC_6_grayscale_f=0.4'
}
data['CNMC_6_grayscale_f=0.5'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_grayscale_f=0.5',
'output_path': 'output_CNMC_6_grayscale_f=0.5'
}
data['CNMC_6_grayscale_f=0.6'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_grayscale_f=0.6',
'output_path': 'output_CNMC_6_grayscale_f=0.6'
}
data['CNMC_6_grayscale_f=0.7'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_grayscale_f=0.7',
'output_path': 'output_CNMC_6_grayscale_f=0.7'
}
data['CNMC_6_grayscale_f=0.8'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_grayscale_f=0.8',
'output_path': 'output_CNMC_6_grayscale_f=0.8'
}
data['CNMC_6_grayscale_f=0.9'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_grayscale_f=0.9',
'output_path': 'output_CNMC_6_grayscale_f=0.9'
}
data['CNMC_6_grayscale_f=1.0'] = {
'classes': ['normal', 'leukemic'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_grayscale_f=1.0',
'output_path': 'output_CNMC_6_grayscale_f=1.0'
}
data['CNMC_6_hsv_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_hsv_f=0.1',
'output_path': 'output_CNMC_6_hsv_f=0.1'
}
data['CNMC_6_hsv_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_hsv_f=0.2',
'output_path': 'output_CNMC_6_hsv_f=0.2'
}
data['CNMC_6_hsv_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_hsv_f=0.3',
'output_path': 'output_CNMC_6_hsv_f=0.3'
}
data['CNMC_6_hsv_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_hsv_f=0.4',
'output_path': 'output_CNMC_6_hsv_f=0.4'
}
data['CNMC_6_hsv_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_hsv_f=0.5',
'output_path': 'output_CNMC_6_hsv_f=0.5'
}
data['CNMC_6_hsv_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_hsv_f=0.6',
'output_path': 'output_CNMC_6_hsv_f=0.6'
}
data['CNMC_6_hsv_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_hsv_f=0.7',
'output_path': 'output_CNMC_6_hsv_f=0.7'
}
data['CNMC_6_hsv_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_hsv_f=0.8',
'output_path': 'output_CNMC_6_hsv_f=0.8'
}
data['CNMC_6_hsv_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_hsv_f=0.9',
'output_path': 'output_CNMC_6_hsv_f=0.9'
}
data['CNMC_6_hsv_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_hsv_f=1.0',
'output_path': 'output_CNMC_6_hsv_f=1.0'
}
data['CNMC_6_blur_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_blur_f=0.1',
'output_path': 'output_CNMC_6_blur_f=0.1'
}
data['CNMC_6_blur_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_blur_f=0.2',
'output_path': 'output_CNMC_6_blur_f=0.2'
}
data['CNMC_6_blur_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_blur_f=0.3',
'output_path': 'output_CNMC_6_blur_f=0.3'
}
data['CNMC_6_blur_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_blur_f=0.4',
'output_path': 'output_CNMC_6_blur_f=0.4'
}
data['CNMC_6_blur_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_blur_f=0.5',
'output_path': 'output_CNMC_6_blur_f=0.5'
}
data['CNMC_6_blur_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_blur_f=0.6',
'output_path': 'output_CNMC_6_blur_f=0.6'
}
data['CNMC_6_blur_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_blur_f=0.7',
'output_path': 'output_CNMC_6_blur_f=0.7'
}
data['CNMC_6_blur_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_blur_f=0.8',
'output_path': 'output_CNMC_6_blur_f=0.8'
}
data['CNMC_6_blur_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_blur_f=0.9',
'output_path': 'output_CNMC_6_blur_f=0.9'
}
data['CNMC_6_blur_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_blur_f=1.0',
'output_path': 'output_CNMC_6_blur_f=1.0'
}

data['CNMC_6_small_random_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_small_random_f=0.1',
'output_path': 'output_CNMC_6_small_random_f=0.1'
}
data['CNMC_6_small_random_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_small_random_f=0.2',
'output_path': 'output_CNMC_6_small_random_f=0.2'
}
data['CNMC_6_small_random_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_small_random_f=0.3',
'output_path': 'output_CNMC_6_small_random_f=0.3'
}
data['CNMC_6_small_random_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_small_random_f=0.4',
'output_path': 'output_CNMC_6_small_random_f=0.4'
}
data['CNMC_6_small_random_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_small_random_f=0.5',
'output_path': 'output_CNMC_6_small_random_f=0.5'
}
data['CNMC_6_small_random_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_small_random_f=0.6',
'output_path': 'output_CNMC_6_small_random_f=0.6'
}
data['CNMC_6_small_random_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_small_random_f=0.7',
'output_path': 'output_CNMC_6_small_random_f=0.7'
}
data['CNMC_6_small_random_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_small_random_f=0.8',
'output_path': 'output_CNMC_6_small_random_f=0.8'
}
data['CNMC_6_small_random_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_small_random_f=0.9',
'output_path': 'output_CNMC_6_small_random_f=0.9'
}
data['CNMC_6_small_random_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_small_random_f=1.0',
'output_path': 'output_CNMC_6_small_random_f=1.0'
}

data['CNMC_6_small_easy_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_small_easy_f=0.1',
  'output_path': 'output_CNMC_6_small_easy_f=0.1'
  }
data['CNMC_6_small_easy_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_small_easy_f=0.2',
  'output_path': 'output_CNMC_6_small_easy_f=0.2'
  }
data['CNMC_6_small_easy_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_small_easy_f=0.3',
  'output_path': 'output_CNMC_6_small_easy_f=0.3'
  }
data['CNMC_6_small_easy_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_small_easy_f=0.4',
  'output_path': 'output_CNMC_6_small_easy_f=0.4'
  }
data['CNMC_6_small_easy_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_small_easy_f=0.5',
  'output_path': 'output_CNMC_6_small_easy_f=0.5'
  }
data['CNMC_6_small_easy_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_small_easy_f=0.6',
  'output_path': 'output_CNMC_6_small_easy_f=0.6'
  }
data['CNMC_6_small_easy_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_small_easy_f=0.7',
  'output_path': 'output_CNMC_6_small_easy_f=0.7'
  }
data['CNMC_6_small_easy_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_small_easy_f=0.8',
  'output_path': 'output_CNMC_6_small_easy_f=0.8'
  }
data['CNMC_6_small_easy_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_small_easy_f=0.9',
  'output_path': 'output_CNMC_6_small_easy_f=0.9'
  }
data['CNMC_6_small_easy_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_small_easy_f=1.0',
  'output_path': 'output_CNMC_6_small_easy_f=1.0'
  }

data['CNMC_6_small_hard_f=0.1'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_small_hard_f=0.1',
  'output_path': 'output_CNMC_6_small_hard_f=0.1'
  }
data['CNMC_6_small_hard_f=0.2'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_small_hard_f=0.2',
  'output_path': 'output_CNMC_6_small_hard_f=0.2'
  }
data['CNMC_6_small_hard_f=0.3'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_small_hard_f=0.3',
  'output_path': 'output_CNMC_6_small_hard_f=0.3'
  }
data['CNMC_6_small_hard_f=0.4'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_small_hard_f=0.4',
  'output_path': 'output_CNMC_6_small_hard_f=0.4'
  }
data['CNMC_6_small_hard_f=0.5'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_small_hard_f=0.5',
  'output_path': 'output_CNMC_6_small_hard_f=0.5'
  }
data['CNMC_6_small_hard_f=0.6'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_small_hard_f=0.6',
  'output_path': 'output_CNMC_6_small_hard_f=0.6'
  }
data['CNMC_6_small_hard_f=0.7'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_small_hard_f=0.7',
  'output_path': 'output_CNMC_6_small_hard_f=0.7'
  }
data['CNMC_6_small_hard_f=0.8'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_small_hard_f=0.8',
  'output_path': 'output_CNMC_6_small_hard_f=0.8'
  }
data['CNMC_6_small_hard_f=0.9'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_small_hard_f=0.9',
  'output_path': 'output_CNMC_6_small_hard_f=0.9'
  }
data['CNMC_6_small_hard_f=1.0'] = {
  'classes': ['benign', 'malignant'],
  'orig_path': 'CNMC_6',
  'dataset_path': 'dataset_CNMC_6_small_hard_f=1.0',
  'output_path': 'output_CNMC_6_small_hard_f=1.0'
  }

data['CNMC_6_small_clusters_f=0.1'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_small_clusters_f=0.1',
'output_path': 'output_CNMC_6_small_clusters_f=0.1'
}
data['CNMC_6_small_clusters_f=0.2'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_small_clusters_f=0.2',
'output_path': 'output_CNMC_6_small_clusters_f=0.2'
}
data['CNMC_6_small_clusters_f=0.3'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_small_clusters_f=0.3',
'output_path': 'output_CNMC_6_small_clusters_f=0.3'
}
data['CNMC_6_small_clusters_f=0.4'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_small_clusters_f=0.4',
'output_path': 'output_CNMC_6_small_clusters_f=0.4'
}
data['CNMC_6_small_clusters_f=0.5'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_small_clusters_f=0.5',
'output_path': 'output_CNMC_6_small_clusters_f=0.5'
}
data['CNMC_6_small_clusters_f=0.6'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_small_clusters_f=0.6',
'output_path': 'output_CNMC_6_small_clusters_f=0.6'
}
data['CNMC_6_small_clusters_f=0.7'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_small_clusters_f=0.7',
'output_path': 'output_CNMC_6_small_clusters_f=0.7'
}
data['CNMC_6_small_clusters_f=0.8'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_small_clusters_f=0.8',
'output_path': 'output_CNMC_6_small_clusters_f=0.8'
}
data['CNMC_6_small_clusters_f=0.9'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_small_clusters_f=0.9',
'output_path': 'output_CNMC_6_small_clusters_f=0.9'
}
data['CNMC_6_small_clusters_f=1.0'] = {
'classes': ['benign', 'malignant'],
'orig_path': 'CNMC_6',
'dataset_path': 'dataset_CNMC_6_small_clusters_f=1.0',
'output_path': 'output_CNMC_6_small_clusters_f=1.0'
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
datasets = ['ISIC_2', 'ISIC_2_image_rot_f=0.1', 'ISIC_2_image_rot_f=0.2',
        'ISIC_2_image_rot_f=0.3', 'ISIC_2_image_rot_f=0.4', 'ISIC_2_image_rot_f=0.5',
        'ISIC_2_image_rot_f=0.6', 'ISIC_2_image_rot_f=0.7', 'ISIC_2_image_rot_f=0.8',
        'ISIC_2_image_rot_f=0.9', 'ISIC_2_image_rot_f=1.0', 'ISIC_2_image_translation_f=0.1',
        'ISIC_2_image_translation_f=0.2', 'ISIC_2_image_translation_f=0.3', 'ISIC_2_image_translation_f=0.4',
        'ISIC_2_image_translation_f=0.5', 'ISIC_2_image_translation_f=0.6', 'ISIC_2_image_translation_f=0.7',
        'ISIC_2_image_translation_f=0.8', 'ISIC_2_image_translation_f=0.9', 'ISIC_2_image_translation_f=1.0',
        'ISIC_2_image_zoom_f=0.1', 'ISIC_2_image_zoom_f=0.2', 'ISIC_2_image_zoom_f=0.3',
        'ISIC_2_image_zoom_f=0.4', 'ISIC_2_image_zoom_f=0.5', 'ISIC_2_image_zoom_f=0.6',
        'ISIC_2_image_zoom_f=0.7', 'ISIC_2_image_zoom_f=0.8', 'ISIC_2_image_zoom_f=0.9',
        'ISIC_2_image_zoom_f=1.0', 'ISIC_2_add_noise_gaussian_f=0.1', 'ISIC_2_add_noise_gaussian_f=0.2',
        'ISIC_2_add_noise_gaussian_f=0.3', 'ISIC_2_add_noise_gaussian_f=0.4', 'ISIC_2_add_noise_gaussian_f=0.5',
        'ISIC_2_add_noise_gaussian_f=0.6', 'ISIC_2_add_noise_gaussian_f=0.7', 'ISIC_2_add_noise_gaussian_f=0.8',
        'ISIC_2_add_noise_gaussian_f=0.9', 'ISIC_2_add_noise_gaussian_f=1.0', 'ISIC_2_add_noise_poisson_f=0.1',
        'ISIC_2_add_noise_poisson_f=0.2', 'ISIC_2_add_noise_poisson_f=0.3', 'ISIC_2_add_noise_poisson_f=0.4',
        'ISIC_2_add_noise_poisson_f=0.5', 'ISIC_2_add_noise_poisson_f=0.6', 'ISIC_2_add_noise_poisson_f=0.7',
        'ISIC_2_add_noise_poisson_f=0.8', 'ISIC_2_add_noise_poisson_f=0.9', 'ISIC_2_add_noise_poisson_f=1.0',
        'ISIC_2_add_noise_salt_and_pepper_f=0.1', 'ISIC_2_add_noise_salt_and_pepper_f=0.2',
        'ISIC_2_add_noise_salt_and_pepper_f=0.3', 'ISIC_2_add_noise_salt_and_pepper_f=0.4',
        'ISIC_2_add_noise_salt_and_pepper_f=0.5', 'ISIC_2_add_noise_salt_and_pepper_f=0.6',
        'ISIC_2_add_noise_salt_and_pepper_f=0.7', 'ISIC_2_add_noise_salt_and_pepper_f=0.8',
        'ISIC_2_add_noise_salt_and_pepper_f=0.9', 'ISIC_2_add_noise_salt_and_pepper_f=1.0',
        'ISIC_2_add_noise_speckle_f=0.1', 'ISIC_2_add_noise_speckle_f=0.2', 'ISIC_2_add_noise_speckle_f=0.3',
        'ISIC_2_add_noise_speckle_f=0.4', 'ISIC_2_add_noise_speckle_f=0.5', 'ISIC_2_add_noise_speckle_f=0.6',
        'ISIC_2_add_noise_speckle_f=0.7', 'ISIC_2_add_noise_speckle_f=0.8', 'ISIC_2_add_noise_speckle_f=0.9',
        'ISIC_2_add_noise_speckle_f=1.0', 'ISIC_2_imbalance_classes_f=0.1', 'ISIC_2_imbalance_classes_f=0.2',
        'ISIC_2_imbalance_classes_f=0.3', 'ISIC_2_imbalance_classes_f=0.4', 'ISIC_2_imbalance_classes_f=0.5',
        'ISIC_2_imbalance_classes_f=0.6', 'ISIC_2_imbalance_classes_f=0.7', 'ISIC_2_imbalance_classes_f=0.8',
        'ISIC_2_imbalance_classes_f=0.9', 'ISIC_2_imbalance_classes_f=1.0', 'ISIC_2_grayscale_f=0.1',
        'ISIC_2_grayscale_f=0.2', 'ISIC_2_grayscale_f=0.3', 'ISIC_2_grayscale_f=0.4',
        'ISIC_2_grayscale_f=0.5', 'ISIC_2_grayscale_f=0.6', 'ISIC_2_grayscale_f=0.7',
        'ISIC_2_grayscale_f=0.8', 'ISIC_2_grayscale_f=0.9', 'ISIC_2_grayscale_f=1.0',
        'ISIC_2_hsv_f=0.1', 'ISIC_2_hsv_f=0.2', 'ISIC_2_hsv_f=0.3', 'ISIC_2_hsv_f=0.4',
        'ISIC_2_hsv_f=0.5', 'ISIC_2_hsv_f=0.6', 'ISIC_2_hsv_f=0.7',
        'ISIC_2_hsv_f=0.8', 'ISIC_2_hsv_f=0.9', 'ISIC_2_hsv_f=1.0',
        'ISIC_2_blur_f=0.1', 'ISIC_2_blur_f=0.2', 'ISIC_2_blur_f=0.3', 'ISIC_2_blur_f=0.4',
        'ISIC_2_blur_f=0.5', 'ISIC_2_blur_f=0.6', 'ISIC_2_blur_f=0.7',
        'ISIC_2_blur_f=0.8', 'ISIC_2_blur_f=0.9', 'ISIC_2_blur_f=1.0',
        'ISIC_2_small_random_f=0.1', 'ISIC_2_small_random_f=0.2', 'ISIC_2_small_random_f=0.3', 'ISIC_2_small_random_f=0.4',
        'ISIC_2_small_random_f=0.5', 'ISIC_2_small_random_f=0.6', 'ISIC_2_small_random_f=0.7',
        'ISIC_2_small_random_f=0.8', 'ISIC_2_small_random_f=0.9', 'ISIC_2_small_random_f=1.0',
        'ISIC_2_small_easy_f=0.1', 'ISIC_2_small_easy_f=0.2', 'ISIC_2_small_easy_f=0.3', 'ISIC_2_small_easy_f=0.4',
        'ISIC_2_small_easy_f=0.5', 'ISIC_2_small_easy_f=0.6', 'ISIC_2_small_easy_f=0.7',
        'ISIC_2_small_easy_f=0.8', 'ISIC_2_small_easy_f=0.9', 'ISIC_2_small_easy_f=1.0',
        'ISIC_2_small_hard_f=0.1', 'ISIC_2_small_hard_f=0.2', 'ISIC_2_small_hard_f=0.3', 'ISIC_2_small_hard_f=0.4',
        'ISIC_2_small_hard_f=0.5', 'ISIC_2_small_hard_f=0.6', 'ISIC_2_small_hard_f=0.7',
        'ISIC_2_small_hard_f=0.8', 'ISIC_2_small_hard_f=0.9', 'ISIC_2_small_hard_f=1.0',
        'ISIC_2_small_clusters_f=0.1', 'ISIC_2_small_clusters_f=0.2', 'ISIC_2_small_clusters_f=0.3', 'ISIC_2_small_clusters_f=0.4',
        'ISIC_2_small_clusters_f=0.5', 'ISIC_2_small_clusters_f=0.6', 'ISIC_2_small_clusters_f=0.7',
        'ISIC_2_small_clusters_f=0.8', 'ISIC_2_small_clusters_f=0.9', 'ISIC_2_small_clusters_f=1.0',
        'ISIC_3', 'ISIC_3_image_rot_f=0.1', 'ISIC_3_image_rot_f=0.2',
        'ISIC_3_image_rot_f=0.3', 'ISIC_3_image_rot_f=0.4', 'ISIC_3_image_rot_f=0.5',
        'ISIC_3_image_rot_f=0.6', 'ISIC_3_image_rot_f=0.7', 'ISIC_3_image_rot_f=0.8',
        'ISIC_3_image_rot_f=0.9', 'ISIC_3_image_rot_f=1.0', 'ISIC_3_image_translation_f=0.1',
        'ISIC_3_image_translation_f=0.2', 'ISIC_3_image_translation_f=0.3', 'ISIC_3_image_translation_f=0.4',
        'ISIC_3_image_translation_f=0.5', 'ISIC_3_image_translation_f=0.6', 'ISIC_3_image_translation_f=0.7',
        'ISIC_3_image_translation_f=0.8', 'ISIC_3_image_translation_f=0.9', 'ISIC_3_image_translation_f=1.0',
        'ISIC_3_image_zoom_f=0.1', 'ISIC_3_image_zoom_f=0.2', 'ISIC_3_image_zoom_f=0.3',
        'ISIC_3_image_zoom_f=0.4', 'ISIC_3_image_zoom_f=0.5', 'ISIC_3_image_zoom_f=0.6',
        'ISIC_3_image_zoom_f=0.7', 'ISIC_3_image_zoom_f=0.8', 'ISIC_3_image_zoom_f=0.9',
        'ISIC_3_image_zoom_f=1.0', 'ISIC_3_add_noise_gaussian_f=0.1', 'ISIC_3_add_noise_gaussian_f=0.2',
        'ISIC_3_add_noise_gaussian_f=0.3', 'ISIC_3_add_noise_gaussian_f=0.4', 'ISIC_3_add_noise_gaussian_f=0.5',
        'ISIC_3_add_noise_gaussian_f=0.6', 'ISIC_3_add_noise_gaussian_f=0.7', 'ISIC_3_add_noise_gaussian_f=0.8',
        'ISIC_3_add_noise_gaussian_f=0.9', 'ISIC_3_add_noise_gaussian_f=1.0', 'ISIC_3_add_noise_poisson_f=0.1',
        'ISIC_3_add_noise_poisson_f=0.2', 'ISIC_3_add_noise_poisson_f=0.3', 'ISIC_3_add_noise_poisson_f=0.4',
        'ISIC_3_add_noise_poisson_f=0.5', 'ISIC_3_add_noise_poisson_f=0.6', 'ISIC_3_add_noise_poisson_f=0.7',
        'ISIC_3_add_noise_poisson_f=0.8', 'ISIC_3_add_noise_poisson_f=0.9', 'ISIC_3_add_noise_poisson_f=1.0',
        'ISIC_3_add_noise_salt_and_pepper_f=0.1', 'ISIC_3_add_noise_salt_and_pepper_f=0.2',
        'ISIC_3_add_noise_salt_and_pepper_f=0.3', 'ISIC_3_add_noise_salt_and_pepper_f=0.4',
        'ISIC_3_add_noise_salt_and_pepper_f=0.5', 'ISIC_3_add_noise_salt_and_pepper_f=0.6',
        'ISIC_3_add_noise_salt_and_pepper_f=0.7', 'ISIC_3_add_noise_salt_and_pepper_f=0.8',
        'ISIC_3_add_noise_salt_and_pepper_f=0.9', 'ISIC_3_add_noise_salt_and_pepper_f=1.0',
        'ISIC_3_add_noise_speckle_f=0.1', 'ISIC_3_add_noise_speckle_f=0.2', 'ISIC_3_add_noise_speckle_f=0.3',
        'ISIC_3_add_noise_speckle_f=0.4', 'ISIC_3_add_noise_speckle_f=0.5', 'ISIC_3_add_noise_speckle_f=0.6',
        'ISIC_3_add_noise_speckle_f=0.7', 'ISIC_3_add_noise_speckle_f=0.8', 'ISIC_3_add_noise_speckle_f=0.9',
        'ISIC_3_add_noise_speckle_f=1.0', 'ISIC_3_imbalance_classes_f=0.1', 'ISIC_3_imbalance_classes_f=0.2',
        'ISIC_3_imbalance_classes_f=0.3', 'ISIC_3_imbalance_classes_f=0.4', 'ISIC_3_imbalance_classes_f=0.5',
        'ISIC_3_imbalance_classes_f=0.6', 'ISIC_3_imbalance_classes_f=0.7', 'ISIC_3_imbalance_classes_f=0.8',
        'ISIC_3_imbalance_classes_f=0.9', 'ISIC_3_imbalance_classes_f=1.0', 'ISIC_3_grayscale_f=0.1',
        'ISIC_3_grayscale_f=0.2', 'ISIC_3_grayscale_f=0.3', 'ISIC_3_grayscale_f=0.4',
        'ISIC_3_grayscale_f=0.5', 'ISIC_3_grayscale_f=0.6', 'ISIC_3_grayscale_f=0.7',
        'ISIC_3_grayscale_f=0.8', 'ISIC_3_grayscale_f=0.9', 'ISIC_3_grayscale_f=1.0',
        'ISIC_3_hsv_f=0.1', 'ISIC_3_hsv_f=0.2', 'ISIC_3_hsv_f=0.3', 'ISIC_3_hsv_f=0.4',
        'ISIC_3_hsv_f=0.5', 'ISIC_3_hsv_f=0.6', 'ISIC_3_hsv_f=0.7',
        'ISIC_3_hsv_f=0.8', 'ISIC_3_hsv_f=0.9', 'ISIC_3_hsv_f=1.0',
        'ISIC_3_blur_f=0.1', 'ISIC_3_blur_f=0.2', 'ISIC_3_blur_f=0.3', 'ISIC_3_blur_f=0.4',
        'ISIC_3_blur_f=0.5', 'ISIC_3_blur_f=0.6', 'ISIC_3_blur_f=0.7',
        'ISIC_3_blur_f=0.8', 'ISIC_3_blur_f=0.9', 'ISIC_3_blur_f=1.0',
        'ISIC_3_small_random_f=0.1', 'ISIC_3_small_random_f=0.2', 'ISIC_3_small_random_f=0.3', 'ISIC_3_small_random_f=0.4',
        'ISIC_3_small_random_f=0.5', 'ISIC_3_small_random_f=0.6', 'ISIC_3_small_random_f=0.7',
        'ISIC_3_small_random_f=0.8', 'ISIC_3_small_random_f=0.9', 'ISIC_3_small_random_f=1.0',
        'ISIC_3_small_easy_f=0.1', 'ISIC_3_small_easy_f=0.2', 'ISIC_3_small_easy_f=0.3', 'ISIC_3_small_easy_f=0.4',
        'ISIC_3_small_easy_f=0.5', 'ISIC_3_small_easy_f=0.6', 'ISIC_3_small_easy_f=0.7',
        'ISIC_3_small_easy_f=0.8', 'ISIC_3_small_easy_f=0.9', 'ISIC_3_small_easy_f=1.0',
        'ISIC_3_small_hard_f=0.1', 'ISIC_3_small_hard_f=0.2', 'ISIC_3_small_hard_f=0.3', 'ISIC_3_small_hard_f=0.4',
        'ISIC_3_small_hard_f=0.5', 'ISIC_3_small_hard_f=0.6', 'ISIC_3_small_hard_f=0.7',
        'ISIC_3_small_hard_f=0.8', 'ISIC_3_small_hard_f=0.9', 'ISIC_3_small_hard_f=1.0',
        'ISIC_3_small_clusters_f=0.1', 'ISIC_3_small_clusters_f=0.2', 'ISIC_3_small_clusters_f=0.3', 'ISIC_3_small_clusters_f=0.4',
        'ISIC_3_small_clusters_f=0.5', 'ISIC_3_small_clusters_f=0.6', 'ISIC_3_small_clusters_f=0.7',
        'ISIC_3_small_clusters_f=0.8', 'ISIC_3_small_clusters_f=0.9', 'ISIC_3_small_clusters_f=1.0',
        'ISIC_4', 'ISIC_4_image_rot_f=0.1', 'ISIC_4_image_rot_f=0.2',
        'ISIC_4_image_rot_f=0.3', 'ISIC_4_image_rot_f=0.4', 'ISIC_4_image_rot_f=0.5',
        'ISIC_4_image_rot_f=0.6', 'ISIC_4_image_rot_f=0.7', 'ISIC_4_image_rot_f=0.8',
        'ISIC_4_image_rot_f=0.9', 'ISIC_4_image_rot_f=1.0', 'ISIC_4_image_translation_f=0.1',
        'ISIC_4_image_translation_f=0.2', 'ISIC_4_image_translation_f=0.3', 'ISIC_4_image_translation_f=0.4',
        'ISIC_4_image_translation_f=0.5', 'ISIC_4_image_translation_f=0.6', 'ISIC_4_image_translation_f=0.7',
        'ISIC_4_image_translation_f=0.8', 'ISIC_4_image_translation_f=0.9', 'ISIC_4_image_translation_f=1.0',
        'ISIC_4_image_zoom_f=0.1', 'ISIC_4_image_zoom_f=0.2', 'ISIC_4_image_zoom_f=0.3',
        'ISIC_4_image_zoom_f=0.4', 'ISIC_4_image_zoom_f=0.5', 'ISIC_4_image_zoom_f=0.6',
        'ISIC_4_image_zoom_f=0.7', 'ISIC_4_image_zoom_f=0.8', 'ISIC_4_image_zoom_f=0.9',
        'ISIC_4_image_zoom_f=1.0', 'ISIC_4_add_noise_gaussian_f=0.1', 'ISIC_4_add_noise_gaussian_f=0.2',
        'ISIC_4_add_noise_gaussian_f=0.3', 'ISIC_4_add_noise_gaussian_f=0.4', 'ISIC_4_add_noise_gaussian_f=0.5',
        'ISIC_4_add_noise_gaussian_f=0.6', 'ISIC_4_add_noise_gaussian_f=0.7', 'ISIC_4_add_noise_gaussian_f=0.8',
        'ISIC_4_add_noise_gaussian_f=0.9', 'ISIC_4_add_noise_gaussian_f=1.0', 'ISIC_4_add_noise_poisson_f=0.1',
        'ISIC_4_add_noise_poisson_f=0.2', 'ISIC_4_add_noise_poisson_f=0.3', 'ISIC_4_add_noise_poisson_f=0.4',
        'ISIC_4_add_noise_poisson_f=0.5', 'ISIC_4_add_noise_poisson_f=0.6', 'ISIC_4_add_noise_poisson_f=0.7',
        'ISIC_4_add_noise_poisson_f=0.8', 'ISIC_4_add_noise_poisson_f=0.9', 'ISIC_4_add_noise_poisson_f=1.0',
        'ISIC_4_add_noise_salt_and_pepper_f=0.1', 'ISIC_4_add_noise_salt_and_pepper_f=0.2',
        'ISIC_4_add_noise_salt_and_pepper_f=0.3', 'ISIC_4_add_noise_salt_and_pepper_f=0.4',
        'ISIC_4_add_noise_salt_and_pepper_f=0.5', 'ISIC_4_add_noise_salt_and_pepper_f=0.6',
        'ISIC_4_add_noise_salt_and_pepper_f=0.7', 'ISIC_4_add_noise_salt_and_pepper_f=0.8',
        'ISIC_4_add_noise_salt_and_pepper_f=0.9', 'ISIC_4_add_noise_salt_and_pepper_f=1.0',
        'ISIC_4_add_noise_speckle_f=0.1', 'ISIC_4_add_noise_speckle_f=0.2', 'ISIC_4_add_noise_speckle_f=0.3',
        'ISIC_4_add_noise_speckle_f=0.4', 'ISIC_4_add_noise_speckle_f=0.5', 'ISIC_4_add_noise_speckle_f=0.6',
        'ISIC_4_add_noise_speckle_f=0.7', 'ISIC_4_add_noise_speckle_f=0.8', 'ISIC_4_add_noise_speckle_f=0.9',
        'ISIC_4_add_noise_speckle_f=1.0', 'ISIC_4_imbalance_classes_f=0.1', 'ISIC_4_imbalance_classes_f=0.2',
        'ISIC_4_imbalance_classes_f=0.3', 'ISIC_4_imbalance_classes_f=0.4', 'ISIC_4_imbalance_classes_f=0.5',
        'ISIC_4_imbalance_classes_f=0.6', 'ISIC_4_imbalance_classes_f=0.7', 'ISIC_4_imbalance_classes_f=0.8',
        'ISIC_4_imbalance_classes_f=0.9', 'ISIC_4_imbalance_classes_f=1.0', 'ISIC_4_grayscale_f=0.1',
        'ISIC_4_grayscale_f=0.2', 'ISIC_4_grayscale_f=0.3', 'ISIC_4_grayscale_f=0.4',
        'ISIC_4_grayscale_f=0.5', 'ISIC_4_grayscale_f=0.6', 'ISIC_4_grayscale_f=0.7',
        'ISIC_4_grayscale_f=0.8', 'ISIC_4_grayscale_f=0.9', 'ISIC_4_grayscale_f=1.0',
        'ISIC_4_hsv_f=0.1', 'ISIC_4_hsv_f=0.2', 'ISIC_4_hsv_f=0.3', 'ISIC_4_hsv_f=0.4',
        'ISIC_4_hsv_f=0.5', 'ISIC_4_hsv_f=0.6', 'ISIC_4_hsv_f=0.7',
        'ISIC_4_hsv_f=0.8', 'ISIC_4_hsv_f=0.9', 'ISIC_4_hsv_f=1.0',
        'ISIC_4_blur_f=0.1', 'ISIC_4_blur_f=0.2', 'ISIC_4_blur_f=0.3', 'ISIC_4_blur_f=0.4',
        'ISIC_4_blur_f=0.5', 'ISIC_4_blur_f=0.6', 'ISIC_4_blur_f=0.7',
        'ISIC_4_blur_f=0.8', 'ISIC_4_blur_f=0.9', 'ISIC_4_blur_f=1.0',
        'ISIC_4_small_random_f=0.1', 'ISIC_4_small_random_f=0.2', 'ISIC_4_small_random_f=0.3', 'ISIC_4_small_random_f=0.4',
        'ISIC_4_small_random_f=0.5', 'ISIC_4_small_random_f=0.6', 'ISIC_4_small_random_f=0.7',
        'ISIC_4_small_random_f=0.8', 'ISIC_4_small_random_f=0.9', 'ISIC_4_small_random_f=1.0',
        'ISIC_4_small_easy_f=0.1', 'ISIC_4_small_easy_f=0.2', 'ISIC_4_small_easy_f=0.3', 'ISIC_4_small_easy_f=0.4',
        'ISIC_4_small_easy_f=0.5', 'ISIC_4_small_easy_f=0.6', 'ISIC_4_small_easy_f=0.7',
        'ISIC_4_small_easy_f=0.8', 'ISIC_4_small_easy_f=0.9', 'ISIC_4_small_easy_f=1.0',
        'ISIC_4_small_hard_f=0.1', 'ISIC_4_small_hard_f=0.2', 'ISIC_4_small_hard_f=0.3', 'ISIC_4_small_hard_f=0.4',
        'ISIC_4_small_hard_f=0.5', 'ISIC_4_small_hard_f=0.6', 'ISIC_4_small_hard_f=0.7',
        'ISIC_4_small_hard_f=0.8', 'ISIC_4_small_hard_f=0.9', 'ISIC_4_small_hard_f=1.0',
        'ISIC_4_small_clusters_f=0.1', 'ISIC_4_small_clusters_f=0.2', 'ISIC_4_small_clusters_f=0.3', 'ISIC_4_small_clusters_f=0.4',
        'ISIC_4_small_clusters_f=0.5', 'ISIC_4_small_clusters_f=0.6', 'ISIC_4_small_clusters_f=0.7',
        'ISIC_4_small_clusters_f=0.8', 'ISIC_4_small_clusters_f=0.9', 'ISIC_4_small_clusters_f=1.0',
        'ISIC_5', 'ISIC_5_image_rot_f=0.1', 'ISIC_5_image_rot_f=0.2',
        'ISIC_5_image_rot_f=0.3', 'ISIC_5_image_rot_f=0.4', 'ISIC_5_image_rot_f=0.5',
        'ISIC_5_image_rot_f=0.6', 'ISIC_5_image_rot_f=0.7', 'ISIC_5_image_rot_f=0.8',
        'ISIC_5_image_rot_f=0.9', 'ISIC_5_image_rot_f=1.0', 'ISIC_5_image_translation_f=0.1',
        'ISIC_5_image_translation_f=0.2', 'ISIC_5_image_translation_f=0.3', 'ISIC_5_image_translation_f=0.4',
        'ISIC_5_image_translation_f=0.5', 'ISIC_5_image_translation_f=0.6', 'ISIC_5_image_translation_f=0.7',
        'ISIC_5_image_translation_f=0.8', 'ISIC_5_image_translation_f=0.9', 'ISIC_5_image_translation_f=1.0',
        'ISIC_5_image_zoom_f=0.1', 'ISIC_5_image_zoom_f=0.2', 'ISIC_5_image_zoom_f=0.3',
        'ISIC_5_image_zoom_f=0.4', 'ISIC_5_image_zoom_f=0.5', 'ISIC_5_image_zoom_f=0.6',
        'ISIC_5_image_zoom_f=0.7', 'ISIC_5_image_zoom_f=0.8', 'ISIC_5_image_zoom_f=0.9',
        'ISIC_5_image_zoom_f=1.0', 'ISIC_5_add_noise_gaussian_f=0.1', 'ISIC_5_add_noise_gaussian_f=0.2',
        'ISIC_5_add_noise_gaussian_f=0.3', 'ISIC_5_add_noise_gaussian_f=0.4', 'ISIC_5_add_noise_gaussian_f=0.5',
        'ISIC_5_add_noise_gaussian_f=0.6', 'ISIC_5_add_noise_gaussian_f=0.7', 'ISIC_5_add_noise_gaussian_f=0.8',
        'ISIC_5_add_noise_gaussian_f=0.9', 'ISIC_5_add_noise_gaussian_f=1.0', 'ISIC_5_add_noise_poisson_f=0.1',
        'ISIC_5_add_noise_poisson_f=0.2', 'ISIC_5_add_noise_poisson_f=0.3', 'ISIC_5_add_noise_poisson_f=0.4',
        'ISIC_5_add_noise_poisson_f=0.5', 'ISIC_5_add_noise_poisson_f=0.6', 'ISIC_5_add_noise_poisson_f=0.7',
        'ISIC_5_add_noise_poisson_f=0.8', 'ISIC_5_add_noise_poisson_f=0.9', 'ISIC_5_add_noise_poisson_f=1.0',
        'ISIC_5_add_noise_salt_and_pepper_f=0.1', 'ISIC_5_add_noise_salt_and_pepper_f=0.2',
        'ISIC_5_add_noise_salt_and_pepper_f=0.3', 'ISIC_5_add_noise_salt_and_pepper_f=0.4',
        'ISIC_5_add_noise_salt_and_pepper_f=0.5', 'ISIC_5_add_noise_salt_and_pepper_f=0.6',
        'ISIC_5_add_noise_salt_and_pepper_f=0.7', 'ISIC_5_add_noise_salt_and_pepper_f=0.8',
        'ISIC_5_add_noise_salt_and_pepper_f=0.9', 'ISIC_5_add_noise_salt_and_pepper_f=1.0',
        'ISIC_5_add_noise_speckle_f=0.1', 'ISIC_5_add_noise_speckle_f=0.2', 'ISIC_5_add_noise_speckle_f=0.3',
        'ISIC_5_add_noise_speckle_f=0.4', 'ISIC_5_add_noise_speckle_f=0.5', 'ISIC_5_add_noise_speckle_f=0.6',
        'ISIC_5_add_noise_speckle_f=0.7', 'ISIC_5_add_noise_speckle_f=0.8', 'ISIC_5_add_noise_speckle_f=0.9',
        'ISIC_5_add_noise_speckle_f=1.0', 'ISIC_5_imbalance_classes_f=0.1', 'ISIC_5_imbalance_classes_f=0.2',
        'ISIC_5_imbalance_classes_f=0.3', 'ISIC_5_imbalance_classes_f=0.4', 'ISIC_5_imbalance_classes_f=0.5',
        'ISIC_5_imbalance_classes_f=0.6', 'ISIC_5_imbalance_classes_f=0.7', 'ISIC_5_imbalance_classes_f=0.8',
        'ISIC_5_imbalance_classes_f=0.9', 'ISIC_5_imbalance_classes_f=1.0', 'ISIC_5_grayscale_f=0.1',
        'ISIC_5_grayscale_f=0.2', 'ISIC_5_grayscale_f=0.3', 'ISIC_5_grayscale_f=0.4',
        'ISIC_5_grayscale_f=0.5', 'ISIC_5_grayscale_f=0.6', 'ISIC_5_grayscale_f=0.7',
        'ISIC_5_grayscale_f=0.8', 'ISIC_5_grayscale_f=0.9', 'ISIC_5_grayscale_f=1.0',
        'ISIC_5_hsv_f=0.1', 'ISIC_5_hsv_f=0.2', 'ISIC_5_hsv_f=0.3', 'ISIC_5_hsv_f=0.4',
        'ISIC_5_hsv_f=0.5', 'ISIC_5_hsv_f=0.6', 'ISIC_5_hsv_f=0.7',
        'ISIC_5_hsv_f=0.8', 'ISIC_5_hsv_f=0.9', 'ISIC_5_hsv_f=1.0',
        'ISIC_5_blur_f=0.1', 'ISIC_5_blur_f=0.2', 'ISIC_5_blur_f=0.3', 'ISIC_5_blur_f=0.4',
        'ISIC_5_blur_f=0.5', 'ISIC_5_blur_f=0.6', 'ISIC_5_blur_f=0.7',
        'ISIC_5_blur_f=0.8', 'ISIC_5_blur_f=0.9', 'ISIC_5_blur_f=1.0',
        'ISIC_5_small_random_f=0.1', 'ISIC_5_small_random_f=0.2', 'ISIC_5_small_random_f=0.3', 'ISIC_5_small_random_f=0.4',
        'ISIC_5_small_random_f=0.5', 'ISIC_5_small_random_f=0.6', 'ISIC_5_small_random_f=0.7',
        'ISIC_5_small_random_f=0.8', 'ISIC_5_small_random_f=0.9', 'ISIC_5_small_random_f=1.0',
        'ISIC_5_small_easy_f=0.1', 'ISIC_5_small_easy_f=0.2', 'ISIC_5_small_easy_f=0.3', 'ISIC_5_small_easy_f=0.4',
        'ISIC_5_small_easy_f=0.5', 'ISIC_5_small_easy_f=0.6', 'ISIC_5_small_easy_f=0.7',
        'ISIC_5_small_easy_f=0.8', 'ISIC_5_small_easy_f=0.9', 'ISIC_5_small_easy_f=1.0',
        'ISIC_5_small_hard_f=0.1', 'ISIC_5_small_hard_f=0.2', 'ISIC_5_small_hard_f=0.3', 'ISIC_5_small_hard_f=0.4',
        'ISIC_5_small_hard_f=0.5', 'ISIC_5_small_hard_f=0.6', 'ISIC_5_small_hard_f=0.7',
        'ISIC_5_small_hard_f=0.8', 'ISIC_5_small_hard_f=0.9', 'ISIC_5_small_hard_f=1.0',
        'ISIC_5_small_clusters_f=0.1', 'ISIC_5_small_clusters_f=0.2', 'ISIC_5_small_clusters_f=0.3', 'ISIC_5_small_clusters_f=0.4',
        'ISIC_5_small_clusters_f=0.5', 'ISIC_5_small_clusters_f=0.6', 'ISIC_5_small_clusters_f=0.7',
        'ISIC_5_small_clusters_f=0.8', 'ISIC_5_small_clusters_f=0.9', 'ISIC_5_small_clusters_f=1.0',
        'ISIC_6', 'ISIC_6_image_rot_f=0.1', 'ISIC_6_image_rot_f=0.2',
        'ISIC_6_image_rot_f=0.3', 'ISIC_6_image_rot_f=0.4', 'ISIC_6_image_rot_f=0.5',
        'ISIC_6_image_rot_f=0.6', 'ISIC_6_image_rot_f=0.7', 'ISIC_6_image_rot_f=0.8',
        'ISIC_6_image_rot_f=0.9', 'ISIC_6_image_rot_f=1.0', 'ISIC_6_image_translation_f=0.1',
        'ISIC_6_image_translation_f=0.2', 'ISIC_6_image_translation_f=0.3', 'ISIC_6_image_translation_f=0.4',
        'ISIC_6_image_translation_f=0.5', 'ISIC_6_image_translation_f=0.6', 'ISIC_6_image_translation_f=0.7',
        'ISIC_6_image_translation_f=0.8', 'ISIC_6_image_translation_f=0.9', 'ISIC_6_image_translation_f=1.0',
        'ISIC_6_image_zoom_f=0.1', 'ISIC_6_image_zoom_f=0.2', 'ISIC_6_image_zoom_f=0.3',
        'ISIC_6_image_zoom_f=0.4', 'ISIC_6_image_zoom_f=0.5', 'ISIC_6_image_zoom_f=0.6',
        'ISIC_6_image_zoom_f=0.7', 'ISIC_6_image_zoom_f=0.8', 'ISIC_6_image_zoom_f=0.9',
        'ISIC_6_image_zoom_f=1.0', 'ISIC_6_add_noise_gaussian_f=0.1', 'ISIC_6_add_noise_gaussian_f=0.2',
        'ISIC_6_add_noise_gaussian_f=0.3', 'ISIC_6_add_noise_gaussian_f=0.4', 'ISIC_6_add_noise_gaussian_f=0.5',
        'ISIC_6_add_noise_gaussian_f=0.6', 'ISIC_6_add_noise_gaussian_f=0.7', 'ISIC_6_add_noise_gaussian_f=0.8',
        'ISIC_6_add_noise_gaussian_f=0.9', 'ISIC_6_add_noise_gaussian_f=1.0', 'ISIC_6_add_noise_poisson_f=0.1',
        'ISIC_6_add_noise_poisson_f=0.2', 'ISIC_6_add_noise_poisson_f=0.3', 'ISIC_6_add_noise_poisson_f=0.4',
        'ISIC_6_add_noise_poisson_f=0.5', 'ISIC_6_add_noise_poisson_f=0.6', 'ISIC_6_add_noise_poisson_f=0.7',
        'ISIC_6_add_noise_poisson_f=0.8', 'ISIC_6_add_noise_poisson_f=0.9', 'ISIC_6_add_noise_poisson_f=1.0',
        'ISIC_6_add_noise_salt_and_pepper_f=0.1', 'ISIC_6_add_noise_salt_and_pepper_f=0.2',
        'ISIC_6_add_noise_salt_and_pepper_f=0.3', 'ISIC_6_add_noise_salt_and_pepper_f=0.4',
        'ISIC_6_add_noise_salt_and_pepper_f=0.5', 'ISIC_6_add_noise_salt_and_pepper_f=0.6',
        'ISIC_6_add_noise_salt_and_pepper_f=0.7', 'ISIC_6_add_noise_salt_and_pepper_f=0.8',
        'ISIC_6_add_noise_salt_and_pepper_f=0.9', 'ISIC_6_add_noise_salt_and_pepper_f=1.0',
        'ISIC_6_add_noise_speckle_f=0.1', 'ISIC_6_add_noise_speckle_f=0.2', 'ISIC_6_add_noise_speckle_f=0.3',
        'ISIC_6_add_noise_speckle_f=0.4', 'ISIC_6_add_noise_speckle_f=0.5', 'ISIC_6_add_noise_speckle_f=0.6',
        'ISIC_6_add_noise_speckle_f=0.7', 'ISIC_6_add_noise_speckle_f=0.8', 'ISIC_6_add_noise_speckle_f=0.9',
        'ISIC_6_add_noise_speckle_f=1.0', 'ISIC_6_imbalance_classes_f=0.1', 'ISIC_6_imbalance_classes_f=0.2',
        'ISIC_6_imbalance_classes_f=0.3', 'ISIC_6_imbalance_classes_f=0.4', 'ISIC_6_imbalance_classes_f=0.5',
        'ISIC_6_imbalance_classes_f=0.6', 'ISIC_6_imbalance_classes_f=0.7', 'ISIC_6_imbalance_classes_f=0.8',
        'ISIC_6_imbalance_classes_f=0.9', 'ISIC_6_imbalance_classes_f=1.0', 'ISIC_6_grayscale_f=0.1',
        'ISIC_6_grayscale_f=0.2', 'ISIC_6_grayscale_f=0.3', 'ISIC_6_grayscale_f=0.4',
        'ISIC_6_grayscale_f=0.5', 'ISIC_6_grayscale_f=0.6', 'ISIC_6_grayscale_f=0.7',
        'ISIC_6_grayscale_f=0.8', 'ISIC_6_grayscale_f=0.9', 'ISIC_6_grayscale_f=1.0',
        'ISIC_6_hsv_f=0.1', 'ISIC_6_hsv_f=0.2', 'ISIC_6_hsv_f=0.3', 'ISIC_6_hsv_f=0.4',
        'ISIC_6_hsv_f=0.5', 'ISIC_6_hsv_f=0.6', 'ISIC_6_hsv_f=0.7',
        'ISIC_6_hsv_f=0.8', 'ISIC_6_hsv_f=0.9', 'ISIC_6_hsv_f=1.0',
        'ISIC_6_blur_f=0.1', 'ISIC_6_blur_f=0.2', 'ISIC_6_blur_f=0.3', 'ISIC_6_blur_f=0.4',
        'ISIC_6_blur_f=0.5', 'ISIC_6_blur_f=0.6', 'ISIC_6_blur_f=0.7',
        'ISIC_6_blur_f=0.8', 'ISIC_6_blur_f=0.9', 'ISIC_6_blur_f=1.0',
        'ISIC_6_small_random_f=0.1', 'ISIC_6_small_random_f=0.2', 'ISIC_6_small_random_f=0.3', 'ISIC_6_small_random_f=0.4',
        'ISIC_6_small_random_f=0.5', 'ISIC_6_small_random_f=0.6', 'ISIC_6_small_random_f=0.7',
        'ISIC_6_small_random_f=0.8', 'ISIC_6_small_random_f=0.9', 'ISIC_6_small_random_f=1.0',
        'ISIC_6_small_easy_f=0.1', 'ISIC_6_small_easy_f=0.2', 'ISIC_6_small_easy_f=0.3', 'ISIC_6_small_easy_f=0.4',
        'ISIC_6_small_easy_f=0.5', 'ISIC_6_small_easy_f=0.6', 'ISIC_6_small_easy_f=0.7',
        'ISIC_6_small_easy_f=0.8', 'ISIC_6_small_easy_f=0.9', 'ISIC_6_small_easy_f=1.0',
        'ISIC_6_small_hard_f=0.1', 'ISIC_6_small_hard_f=0.2', 'ISIC_6_small_hard_f=0.3', 'ISIC_6_small_hard_f=0.4',
        'ISIC_6_small_hard_f=0.5', 'ISIC_6_small_hard_f=0.6', 'ISIC_6_small_hard_f=0.7',
        'ISIC_6_small_hard_f=0.8', 'ISIC_6_small_hard_f=0.9', 'ISIC_6_small_hard_f=1.0',
        'ISIC_6_small_clusters_f=0.1', 'ISIC_6_small_clusters_f=0.2', 'ISIC_6_small_clusters_f=0.3', 'ISIC_6_small_clusters_f=0.4',
        'ISIC_6_small_clusters_f=0.5', 'ISIC_6_small_clusters_f=0.6', 'ISIC_6_small_clusters_f=0.7',
        'ISIC_6_small_clusters_f=0.8', 'ISIC_6_small_clusters_f=0.9', 'ISIC_6_small_clusters_f=1.0',
        'CNMC_2', 'CNMC_2_image_rot_f=0.1', 'CNMC_2_image_rot_f=0.2',
        'CNMC_2_image_rot_f=0.3', 'CNMC_2_image_rot_f=0.4', 'CNMC_2_image_rot_f=0.5',
        'CNMC_2_image_rot_f=0.6', 'CNMC_2_image_rot_f=0.7', 'CNMC_2_image_rot_f=0.8',
        'CNMC_2_image_rot_f=0.9', 'CNMC_2_image_rot_f=1.0', 'CNMC_2_image_translation_f=0.1',
        'CNMC_2_image_translation_f=0.2', 'CNMC_2_image_translation_f=0.3', 'CNMC_2_image_translation_f=0.4',
        'CNMC_2_image_translation_f=0.5', 'CNMC_2_image_translation_f=0.6', 'CNMC_2_image_translation_f=0.7',
        'CNMC_2_image_translation_f=0.8', 'CNMC_2_image_translation_f=0.9', 'CNMC_2_image_translation_f=1.0',
        'CNMC_2_image_zoom_f=0.1', 'CNMC_2_image_zoom_f=0.2', 'CNMC_2_image_zoom_f=0.3',
        'CNMC_2_image_zoom_f=0.4', 'CNMC_2_image_zoom_f=0.5', 'CNMC_2_image_zoom_f=0.6',
        'CNMC_2_image_zoom_f=0.7', 'CNMC_2_image_zoom_f=0.8', 'CNMC_2_image_zoom_f=0.9',
        'CNMC_2_image_zoom_f=1.0', 'CNMC_2_add_noise_gaussian_f=0.1', 'CNMC_2_add_noise_gaussian_f=0.2',
        'CNMC_2_add_noise_gaussian_f=0.3', 'CNMC_2_add_noise_gaussian_f=0.4', 'CNMC_2_add_noise_gaussian_f=0.5',
        'CNMC_2_add_noise_gaussian_f=0.6', 'CNMC_2_add_noise_gaussian_f=0.7', 'CNMC_2_add_noise_gaussian_f=0.8',
        'CNMC_2_add_noise_gaussian_f=0.9', 'CNMC_2_add_noise_gaussian_f=1.0', 'CNMC_2_add_noise_poisson_f=0.1',
        'CNMC_2_add_noise_poisson_f=0.2', 'CNMC_2_add_noise_poisson_f=0.3', 'CNMC_2_add_noise_poisson_f=0.4',
        'CNMC_2_add_noise_poisson_f=0.5', 'CNMC_2_add_noise_poisson_f=0.6', 'CNMC_2_add_noise_poisson_f=0.7',
        'CNMC_2_add_noise_poisson_f=0.8', 'CNMC_2_add_noise_poisson_f=0.9', 'CNMC_2_add_noise_poisson_f=1.0',
        'CNMC_2_add_noise_salt_and_pepper_f=0.1', 'CNMC_2_add_noise_salt_and_pepper_f=0.2',
        'CNMC_2_add_noise_salt_and_pepper_f=0.3', 'CNMC_2_add_noise_salt_and_pepper_f=0.4',
        'CNMC_2_add_noise_salt_and_pepper_f=0.5', 'CNMC_2_add_noise_salt_and_pepper_f=0.6',
        'CNMC_2_add_noise_salt_and_pepper_f=0.7', 'CNMC_2_add_noise_salt_and_pepper_f=0.8',
        'CNMC_2_add_noise_salt_and_pepper_f=0.9', 'CNMC_2_add_noise_salt_and_pepper_f=1.0',
        'CNMC_2_add_noise_speckle_f=0.1', 'CNMC_2_add_noise_speckle_f=0.2', 'CNMC_2_add_noise_speckle_f=0.3',
        'CNMC_2_add_noise_speckle_f=0.4', 'CNMC_2_add_noise_speckle_f=0.5', 'CNMC_2_add_noise_speckle_f=0.6',
        'CNMC_2_add_noise_speckle_f=0.7', 'CNMC_2_add_noise_speckle_f=0.8', 'CNMC_2_add_noise_speckle_f=0.9',
        'CNMC_2_add_noise_speckle_f=1.0', 'CNMC_2_imbalance_classes_f=0.1', 'CNMC_2_imbalance_classes_f=0.2',
        'CNMC_2_imbalance_classes_f=0.3', 'CNMC_2_imbalance_classes_f=0.4', 'CNMC_2_imbalance_classes_f=0.5',
        'CNMC_2_imbalance_classes_f=0.6', 'CNMC_2_imbalance_classes_f=0.7', 'CNMC_2_imbalance_classes_f=0.8',
        'CNMC_2_imbalance_classes_f=0.9', 'CNMC_2_imbalance_classes_f=1.0', 'CNMC_2_grayscale_f=0.1',
        'CNMC_2_grayscale_f=0.2', 'CNMC_2_grayscale_f=0.3', 'CNMC_2_grayscale_f=0.4',
        'CNMC_2_grayscale_f=0.5', 'CNMC_2_grayscale_f=0.6', 'CNMC_2_grayscale_f=0.7',
        'CNMC_2_grayscale_f=0.8', 'CNMC_2_grayscale_f=0.9', 'CNMC_2_grayscale_f=1.0',
        'CNMC_2_hsv_f=0.1', 'CNMC_2_hsv_f=0.2', 'CNMC_2_hsv_f=0.3', 'CNMC_2_hsv_f=0.4',
        'CNMC_2_hsv_f=0.5', 'CNMC_2_hsv_f=0.6', 'CNMC_2_hsv_f=0.7',
        'CNMC_2_hsv_f=0.8', 'CNMC_2_hsv_f=0.9', 'CNMC_2_hsv_f=1.0',
        'CNMC_2_blur_f=0.1', 'CNMC_2_blur_f=0.2', 'CNMC_2_blur_f=0.3', 'CNMC_2_blur_f=0.4',
        'CNMC_2_blur_f=0.5', 'CNMC_2_blur_f=0.6', 'CNMC_2_blur_f=0.7',
        'CNMC_2_blur_f=0.8', 'CNMC_2_blur_f=0.9', 'CNMC_2_blur_f=1.0',
        'CNMC_2_small_random_f=0.1', 'CNMC_2_small_random_f=0.2', 'CNMC_2_small_random_f=0.3', 'CNMC_2_small_random_f=0.4',
        'CNMC_2_small_random_f=0.5', 'CNMC_2_small_random_f=0.6', 'CNMC_2_small_random_f=0.7',
        'CNMC_2_small_random_f=0.8', 'CNMC_2_small_random_f=0.9', 'CNMC_2_small_random_f=1.0',
        'CNMC_2_small_easy_f=0.1', 'CNMC_2_small_easy_f=0.2', 'CNMC_2_small_easy_f=0.3', 'CNMC_2_small_easy_f=0.4',
        'CNMC_2_small_easy_f=0.5', 'CNMC_2_small_easy_f=0.6', 'CNMC_2_small_easy_f=0.7',
        'CNMC_2_small_easy_f=0.8', 'CNMC_2_small_easy_f=0.9', 'CNMC_2_small_easy_f=1.0',
        'CNMC_2_small_hard_f=0.1', 'CNMC_2_small_hard_f=0.2', 'CNMC_2_small_hard_f=0.3', 'CNMC_2_small_hard_f=0.4',
        'CNMC_2_small_hard_f=0.5', 'CNMC_2_small_hard_f=0.6', 'CNMC_2_small_hard_f=0.7',
        'CNMC_2_small_hard_f=0.8', 'CNMC_2_small_hard_f=0.9', 'CNMC_2_small_hard_f=1.0',
        'CNMC_2_small_clusters_f=0.1', 'CNMC_2_small_clusters_f=0.2', 'CNMC_2_small_clusters_f=0.3', 'CNMC_2_small_clusters_f=0.4',
        'CNMC_2_small_clusters_f=0.5', 'CNMC_2_small_clusters_f=0.6', 'CNMC_2_small_clusters_f=0.7',
        'CNMC_2_small_clusters_f=0.8', 'CNMC_2_small_clusters_f=0.9', 'CNMC_2_small_clusters_f=1.0',
        'CNMC_3', 'CNMC_3_image_rot_f=0.1', 'CNMC_3_image_rot_f=0.2',
        'CNMC_3_image_rot_f=0.3', 'CNMC_3_image_rot_f=0.4', 'CNMC_3_image_rot_f=0.5',
        'CNMC_3_image_rot_f=0.6', 'CNMC_3_image_rot_f=0.7', 'CNMC_3_image_rot_f=0.8',
        'CNMC_3_image_rot_f=0.9', 'CNMC_3_image_rot_f=1.0', 'CNMC_3_image_translation_f=0.1',
        'CNMC_3_image_translation_f=0.2', 'CNMC_3_image_translation_f=0.3', 'CNMC_3_image_translation_f=0.4',
        'CNMC_3_image_translation_f=0.5', 'CNMC_3_image_translation_f=0.6', 'CNMC_3_image_translation_f=0.7',
        'CNMC_3_image_translation_f=0.8', 'CNMC_3_image_translation_f=0.9', 'CNMC_3_image_translation_f=1.0',
        'CNMC_3_image_zoom_f=0.1', 'CNMC_3_image_zoom_f=0.2', 'CNMC_3_image_zoom_f=0.3',
        'CNMC_3_image_zoom_f=0.4', 'CNMC_3_image_zoom_f=0.5', 'CNMC_3_image_zoom_f=0.6',
        'CNMC_3_image_zoom_f=0.7', 'CNMC_3_image_zoom_f=0.8', 'CNMC_3_image_zoom_f=0.9',
        'CNMC_3_image_zoom_f=1.0', 'CNMC_3_add_noise_gaussian_f=0.1', 'CNMC_3_add_noise_gaussian_f=0.2',
        'CNMC_3_add_noise_gaussian_f=0.3', 'CNMC_3_add_noise_gaussian_f=0.4', 'CNMC_3_add_noise_gaussian_f=0.5',
        'CNMC_3_add_noise_gaussian_f=0.6', 'CNMC_3_add_noise_gaussian_f=0.7', 'CNMC_3_add_noise_gaussian_f=0.8',
        'CNMC_3_add_noise_gaussian_f=0.9', 'CNMC_3_add_noise_gaussian_f=1.0', 'CNMC_3_add_noise_poisson_f=0.1',
        'CNMC_3_add_noise_poisson_f=0.2', 'CNMC_3_add_noise_poisson_f=0.3', 'CNMC_3_add_noise_poisson_f=0.4',
        'CNMC_3_add_noise_poisson_f=0.5', 'CNMC_3_add_noise_poisson_f=0.6', 'CNMC_3_add_noise_poisson_f=0.7',
        'CNMC_3_add_noise_poisson_f=0.8', 'CNMC_3_add_noise_poisson_f=0.9', 'CNMC_3_add_noise_poisson_f=1.0',
        'CNMC_3_add_noise_salt_and_pepper_f=0.1', 'CNMC_3_add_noise_salt_and_pepper_f=0.2',
        'CNMC_3_add_noise_salt_and_pepper_f=0.3', 'CNMC_3_add_noise_salt_and_pepper_f=0.4',
        'CNMC_3_add_noise_salt_and_pepper_f=0.5', 'CNMC_3_add_noise_salt_and_pepper_f=0.6',
        'CNMC_3_add_noise_salt_and_pepper_f=0.7', 'CNMC_3_add_noise_salt_and_pepper_f=0.8',
        'CNMC_3_add_noise_salt_and_pepper_f=0.9', 'CNMC_3_add_noise_salt_and_pepper_f=1.0',
        'CNMC_3_add_noise_speckle_f=0.1', 'CNMC_3_add_noise_speckle_f=0.2', 'CNMC_3_add_noise_speckle_f=0.3',
        'CNMC_3_add_noise_speckle_f=0.4', 'CNMC_3_add_noise_speckle_f=0.5', 'CNMC_3_add_noise_speckle_f=0.6',
        'CNMC_3_add_noise_speckle_f=0.7', 'CNMC_3_add_noise_speckle_f=0.8', 'CNMC_3_add_noise_speckle_f=0.9',
        'CNMC_3_add_noise_speckle_f=1.0', 'CNMC_3_imbalance_classes_f=0.1', 'CNMC_3_imbalance_classes_f=0.2',
        'CNMC_3_imbalance_classes_f=0.3', 'CNMC_3_imbalance_classes_f=0.4', 'CNMC_3_imbalance_classes_f=0.5',
        'CNMC_3_imbalance_classes_f=0.6', 'CNMC_3_imbalance_classes_f=0.7', 'CNMC_3_imbalance_classes_f=0.8',
        'CNMC_3_imbalance_classes_f=0.9', 'CNMC_3_imbalance_classes_f=1.0', 'CNMC_3_grayscale_f=0.1',
        'CNMC_3_grayscale_f=0.2', 'CNMC_3_grayscale_f=0.3', 'CNMC_3_grayscale_f=0.4',
        'CNMC_3_grayscale_f=0.5', 'CNMC_3_grayscale_f=0.6', 'CNMC_3_grayscale_f=0.7',
        'CNMC_3_grayscale_f=0.8', 'CNMC_3_grayscale_f=0.9', 'CNMC_3_grayscale_f=1.0',
        'CNMC_3_hsv_f=0.1', 'CNMC_3_hsv_f=0.2', 'CNMC_3_hsv_f=0.3', 'CNMC_3_hsv_f=0.4',
        'CNMC_3_hsv_f=0.5', 'CNMC_3_hsv_f=0.6', 'CNMC_3_hsv_f=0.7',
        'CNMC_3_hsv_f=0.8', 'CNMC_3_hsv_f=0.9', 'CNMC_3_hsv_f=1.0',
        'CNMC_3_blur_f=0.1', 'CNMC_3_blur_f=0.2', 'CNMC_3_blur_f=0.3', 'CNMC_3_blur_f=0.4',
        'CNMC_3_blur_f=0.5', 'CNMC_3_blur_f=0.6', 'CNMC_3_blur_f=0.7',
        'CNMC_3_blur_f=0.8', 'CNMC_3_blur_f=0.9', 'CNMC_3_blur_f=1.0',
        'CNMC_3_small_random_f=0.1', 'CNMC_3_small_random_f=0.2', 'CNMC_3_small_random_f=0.3', 'CNMC_3_small_random_f=0.4',
        'CNMC_3_small_random_f=0.5', 'CNMC_3_small_random_f=0.6', 'CNMC_3_small_random_f=0.7',
        'CNMC_3_small_random_f=0.8', 'CNMC_3_small_random_f=0.9', 'CNMC_3_small_random_f=1.0',
        'CNMC_3_small_easy_f=0.1', 'CNMC_3_small_easy_f=0.2', 'CNMC_3_small_easy_f=0.3', 'CNMC_3_small_easy_f=0.4',
        'CNMC_3_small_easy_f=0.5', 'CNMC_3_small_easy_f=0.6', 'CNMC_3_small_easy_f=0.7',
        'CNMC_3_small_easy_f=0.8', 'CNMC_3_small_easy_f=0.9', 'CNMC_3_small_easy_f=1.0',
        'CNMC_3_small_hard_f=0.1', 'CNMC_3_small_hard_f=0.2', 'CNMC_3_small_hard_f=0.3', 'CNMC_3_small_hard_f=0.4',
        'CNMC_3_small_hard_f=0.5', 'CNMC_3_small_hard_f=0.6', 'CNMC_3_small_hard_f=0.7',
        'CNMC_3_small_hard_f=0.8', 'CNMC_3_small_hard_f=0.9', 'CNMC_3_small_hard_f=1.0',
        'CNMC_3_small_clusters_f=0.1', 'CNMC_3_small_clusters_f=0.2', 'CNMC_3_small_clusters_f=0.3', 'CNMC_3_small_clusters_f=0.4',
        'CNMC_3_small_clusters_f=0.5', 'CNMC_3_small_clusters_f=0.6', 'CNMC_3_small_clusters_f=0.7',
        'CNMC_3_small_clusters_f=0.8', 'CNMC_3_small_clusters_f=0.9', 'CNMC_3_small_clusters_f=1.0',
        'CNMC_4', 'CNMC_4_image_rot_f=0.1', 'CNMC_4_image_rot_f=0.2',
        'CNMC_4_image_rot_f=0.3', 'CNMC_4_image_rot_f=0.4', 'CNMC_4_image_rot_f=0.5',
        'CNMC_4_image_rot_f=0.6', 'CNMC_4_image_rot_f=0.7', 'CNMC_4_image_rot_f=0.8',
        'CNMC_4_image_rot_f=0.9', 'CNMC_4_image_rot_f=1.0', 'CNMC_4_image_translation_f=0.1',
        'CNMC_4_image_translation_f=0.2', 'CNMC_4_image_translation_f=0.3', 'CNMC_4_image_translation_f=0.4',
        'CNMC_4_image_translation_f=0.5', 'CNMC_4_image_translation_f=0.6', 'CNMC_4_image_translation_f=0.7',
        'CNMC_4_image_translation_f=0.8', 'CNMC_4_image_translation_f=0.9', 'CNMC_4_image_translation_f=1.0',
        'CNMC_4_image_zoom_f=0.1', 'CNMC_4_image_zoom_f=0.2', 'CNMC_4_image_zoom_f=0.3',
        'CNMC_4_image_zoom_f=0.4', 'CNMC_4_image_zoom_f=0.5', 'CNMC_4_image_zoom_f=0.6',
        'CNMC_4_image_zoom_f=0.7', 'CNMC_4_image_zoom_f=0.8', 'CNMC_4_image_zoom_f=0.9',
        'CNMC_4_image_zoom_f=1.0', 'CNMC_4_add_noise_gaussian_f=0.1', 'CNMC_4_add_noise_gaussian_f=0.2',
        'CNMC_4_add_noise_gaussian_f=0.3', 'CNMC_4_add_noise_gaussian_f=0.4', 'CNMC_4_add_noise_gaussian_f=0.5',
        'CNMC_4_add_noise_gaussian_f=0.6', 'CNMC_4_add_noise_gaussian_f=0.7', 'CNMC_4_add_noise_gaussian_f=0.8',
        'CNMC_4_add_noise_gaussian_f=0.9', 'CNMC_4_add_noise_gaussian_f=1.0', 'CNMC_4_add_noise_poisson_f=0.1',
        'CNMC_4_add_noise_poisson_f=0.2', 'CNMC_4_add_noise_poisson_f=0.3', 'CNMC_4_add_noise_poisson_f=0.4',
        'CNMC_4_add_noise_poisson_f=0.5', 'CNMC_4_add_noise_poisson_f=0.6', 'CNMC_4_add_noise_poisson_f=0.7',
        'CNMC_4_add_noise_poisson_f=0.8', 'CNMC_4_add_noise_poisson_f=0.9', 'CNMC_4_add_noise_poisson_f=1.0',
        'CNMC_4_add_noise_salt_and_pepper_f=0.1', 'CNMC_4_add_noise_salt_and_pepper_f=0.2',
        'CNMC_4_add_noise_salt_and_pepper_f=0.3', 'CNMC_4_add_noise_salt_and_pepper_f=0.4',
        'CNMC_4_add_noise_salt_and_pepper_f=0.5', 'CNMC_4_add_noise_salt_and_pepper_f=0.6',
        'CNMC_4_add_noise_salt_and_pepper_f=0.7', 'CNMC_4_add_noise_salt_and_pepper_f=0.8',
        'CNMC_4_add_noise_salt_and_pepper_f=0.9', 'CNMC_4_add_noise_salt_and_pepper_f=1.0',
        'CNMC_4_add_noise_speckle_f=0.1', 'CNMC_4_add_noise_speckle_f=0.2', 'CNMC_4_add_noise_speckle_f=0.3',
        'CNMC_4_add_noise_speckle_f=0.4', 'CNMC_4_add_noise_speckle_f=0.5', 'CNMC_4_add_noise_speckle_f=0.6',
        'CNMC_4_add_noise_speckle_f=0.7', 'CNMC_4_add_noise_speckle_f=0.8', 'CNMC_4_add_noise_speckle_f=0.9',
        'CNMC_4_add_noise_speckle_f=1.0', 'CNMC_4_imbalance_classes_f=0.1', 'CNMC_4_imbalance_classes_f=0.2',
        'CNMC_4_imbalance_classes_f=0.3', 'CNMC_4_imbalance_classes_f=0.4', 'CNMC_4_imbalance_classes_f=0.5',
        'CNMC_4_imbalance_classes_f=0.6', 'CNMC_4_imbalance_classes_f=0.7', 'CNMC_4_imbalance_classes_f=0.8',
        'CNMC_4_imbalance_classes_f=0.9', 'CNMC_4_imbalance_classes_f=1.0', 'CNMC_4_grayscale_f=0.1',
        'CNMC_4_grayscale_f=0.2', 'CNMC_4_grayscale_f=0.3', 'CNMC_4_grayscale_f=0.4',
        'CNMC_4_grayscale_f=0.5', 'CNMC_4_grayscale_f=0.6', 'CNMC_4_grayscale_f=0.7',
        'CNMC_4_grayscale_f=0.8', 'CNMC_4_grayscale_f=0.9', 'CNMC_4_grayscale_f=1.0',
        'CNMC_4_hsv_f=0.1', 'CNMC_4_hsv_f=0.2', 'CNMC_4_hsv_f=0.3', 'CNMC_4_hsv_f=0.4',
        'CNMC_4_hsv_f=0.5', 'CNMC_4_hsv_f=0.6', 'CNMC_4_hsv_f=0.7',
        'CNMC_4_hsv_f=0.8', 'CNMC_4_hsv_f=0.9', 'CNMC_4_hsv_f=1.0',
        'CNMC_4_blur_f=0.1', 'CNMC_4_blur_f=0.2', 'CNMC_4_blur_f=0.3', 'CNMC_4_blur_f=0.4',
        'CNMC_4_blur_f=0.5', 'CNMC_4_blur_f=0.6', 'CNMC_4_blur_f=0.7',
        'CNMC_4_blur_f=0.8', 'CNMC_4_blur_f=0.9', 'CNMC_4_blur_f=1.0',
        'CNMC_4_small_random_f=0.1', 'CNMC_4_small_random_f=0.2', 'CNMC_4_small_random_f=0.3', 'CNMC_4_small_random_f=0.4',
        'CNMC_4_small_random_f=0.5', 'CNMC_4_small_random_f=0.6', 'CNMC_4_small_random_f=0.7',
        'CNMC_4_small_random_f=0.8', 'CNMC_4_small_random_f=0.9', 'CNMC_4_small_random_f=1.0',
        'CNMC_4_small_easy_f=0.1', 'CNMC_4_small_easy_f=0.2', 'CNMC_4_small_easy_f=0.3', 'CNMC_4_small_easy_f=0.4',
        'CNMC_4_small_easy_f=0.5', 'CNMC_4_small_easy_f=0.6', 'CNMC_4_small_easy_f=0.7',
        'CNMC_4_small_easy_f=0.8', 'CNMC_4_small_easy_f=0.9', 'CNMC_4_small_easy_f=1.0',
        'CNMC_4_small_hard_f=0.1', 'CNMC_4_small_hard_f=0.2', 'CNMC_4_small_hard_f=0.3', 'CNMC_4_small_hard_f=0.4',
        'CNMC_4_small_hard_f=0.5', 'CNMC_4_small_hard_f=0.6', 'CNMC_4_small_hard_f=0.7',
        'CNMC_4_small_hard_f=0.8', 'CNMC_4_small_hard_f=0.9', 'CNMC_4_small_hard_f=1.0',
        'CNMC_4_small_clusters_f=0.1', 'CNMC_4_small_clusters_f=0.2', 'CNMC_4_small_clusters_f=0.3', 'CNMC_4_small_clusters_f=0.4',
        'CNMC_4_small_clusters_f=0.5', 'CNMC_4_small_clusters_f=0.6', 'CNMC_4_small_clusters_f=0.7',
        'CNMC_4_small_clusters_f=0.8', 'CNMC_4_small_clusters_f=0.9', 'CNMC_4_small_clusters_f=1.0',
        'CNMC_5', 'CNMC_5_image_rot_f=0.1', 'CNMC_5_image_rot_f=0.2',
        'CNMC_5_image_rot_f=0.3', 'CNMC_5_image_rot_f=0.4', 'CNMC_5_image_rot_f=0.5',
        'CNMC_5_image_rot_f=0.6', 'CNMC_5_image_rot_f=0.7', 'CNMC_5_image_rot_f=0.8',
        'CNMC_5_image_rot_f=0.9', 'CNMC_5_image_rot_f=1.0', 'CNMC_5_image_translation_f=0.1',
        'CNMC_5_image_translation_f=0.2', 'CNMC_5_image_translation_f=0.3', 'CNMC_5_image_translation_f=0.4',
        'CNMC_5_image_translation_f=0.5', 'CNMC_5_image_translation_f=0.6', 'CNMC_5_image_translation_f=0.7',
        'CNMC_5_image_translation_f=0.8', 'CNMC_5_image_translation_f=0.9', 'CNMC_5_image_translation_f=1.0',
        'CNMC_5_image_zoom_f=0.1', 'CNMC_5_image_zoom_f=0.2', 'CNMC_5_image_zoom_f=0.3',
        'CNMC_5_image_zoom_f=0.4', 'CNMC_5_image_zoom_f=0.5', 'CNMC_5_image_zoom_f=0.6',
        'CNMC_5_image_zoom_f=0.7', 'CNMC_5_image_zoom_f=0.8', 'CNMC_5_image_zoom_f=0.9',
        'CNMC_5_image_zoom_f=1.0', 'CNMC_5_add_noise_gaussian_f=0.1', 'CNMC_5_add_noise_gaussian_f=0.2',
        'CNMC_5_add_noise_gaussian_f=0.3', 'CNMC_5_add_noise_gaussian_f=0.4', 'CNMC_5_add_noise_gaussian_f=0.5',
        'CNMC_5_add_noise_gaussian_f=0.6', 'CNMC_5_add_noise_gaussian_f=0.7', 'CNMC_5_add_noise_gaussian_f=0.8',
        'CNMC_5_add_noise_gaussian_f=0.9', 'CNMC_5_add_noise_gaussian_f=1.0', 'CNMC_5_add_noise_poisson_f=0.1',
        'CNMC_5_add_noise_poisson_f=0.2', 'CNMC_5_add_noise_poisson_f=0.3', 'CNMC_5_add_noise_poisson_f=0.4',
        'CNMC_5_add_noise_poisson_f=0.5', 'CNMC_5_add_noise_poisson_f=0.6', 'CNMC_5_add_noise_poisson_f=0.7',
        'CNMC_5_add_noise_poisson_f=0.8', 'CNMC_5_add_noise_poisson_f=0.9', 'CNMC_5_add_noise_poisson_f=1.0',
        'CNMC_5_add_noise_salt_and_pepper_f=0.1', 'CNMC_5_add_noise_salt_and_pepper_f=0.2',
        'CNMC_5_add_noise_salt_and_pepper_f=0.3', 'CNMC_5_add_noise_salt_and_pepper_f=0.4',
        'CNMC_5_add_noise_salt_and_pepper_f=0.5', 'CNMC_5_add_noise_salt_and_pepper_f=0.6',
        'CNMC_5_add_noise_salt_and_pepper_f=0.7', 'CNMC_5_add_noise_salt_and_pepper_f=0.8',
        'CNMC_5_add_noise_salt_and_pepper_f=0.9', 'CNMC_5_add_noise_salt_and_pepper_f=1.0',
        'CNMC_5_add_noise_speckle_f=0.1', 'CNMC_5_add_noise_speckle_f=0.2', 'CNMC_5_add_noise_speckle_f=0.3',
        'CNMC_5_add_noise_speckle_f=0.4', 'CNMC_5_add_noise_speckle_f=0.5', 'CNMC_5_add_noise_speckle_f=0.6',
        'CNMC_5_add_noise_speckle_f=0.7', 'CNMC_5_add_noise_speckle_f=0.8', 'CNMC_5_add_noise_speckle_f=0.9',
        'CNMC_5_add_noise_speckle_f=1.0', 'CNMC_5_imbalance_classes_f=0.1', 'CNMC_5_imbalance_classes_f=0.2',
        'CNMC_5_imbalance_classes_f=0.3', 'CNMC_5_imbalance_classes_f=0.4', 'CNMC_5_imbalance_classes_f=0.5',
        'CNMC_5_imbalance_classes_f=0.6', 'CNMC_5_imbalance_classes_f=0.7', 'CNMC_5_imbalance_classes_f=0.8',
        'CNMC_5_imbalance_classes_f=0.9', 'CNMC_5_imbalance_classes_f=1.0', 'CNMC_5_grayscale_f=0.1',
        'CNMC_5_grayscale_f=0.2', 'CNMC_5_grayscale_f=0.3', 'CNMC_5_grayscale_f=0.4',
        'CNMC_5_grayscale_f=0.5', 'CNMC_5_grayscale_f=0.6', 'CNMC_5_grayscale_f=0.7',
        'CNMC_5_grayscale_f=0.8', 'CNMC_5_grayscale_f=0.9', 'CNMC_5_grayscale_f=1.0',
        'CNMC_5_hsv_f=0.1', 'CNMC_5_hsv_f=0.2', 'CNMC_5_hsv_f=0.3', 'CNMC_5_hsv_f=0.4',
        'CNMC_5_hsv_f=0.5', 'CNMC_5_hsv_f=0.6', 'CNMC_5_hsv_f=0.7',
        'CNMC_5_hsv_f=0.8', 'CNMC_5_hsv_f=0.9', 'CNMC_5_hsv_f=1.0',
        'CNMC_5_blur_f=0.1', 'CNMC_5_blur_f=0.2', 'CNMC_5_blur_f=0.3', 'CNMC_5_blur_f=0.4',
        'CNMC_5_blur_f=0.5', 'CNMC_5_blur_f=0.6', 'CNMC_5_blur_f=0.7',
        'CNMC_5_blur_f=0.8', 'CNMC_5_blur_f=0.9', 'CNMC_5_blur_f=1.0',
        'CNMC_5_small_random_f=0.1', 'CNMC_5_small_random_f=0.2', 'CNMC_5_small_random_f=0.3', 'CNMC_5_small_random_f=0.4',
        'CNMC_5_small_random_f=0.5', 'CNMC_5_small_random_f=0.6', 'CNMC_5_small_random_f=0.7',
        'CNMC_5_small_random_f=0.8', 'CNMC_5_small_random_f=0.9', 'CNMC_5_small_random_f=1.0',
        'CNMC_5_small_easy_f=0.1', 'CNMC_5_small_easy_f=0.2', 'CNMC_5_small_easy_f=0.3', 'CNMC_5_small_easy_f=0.4',
        'CNMC_5_small_easy_f=0.5', 'CNMC_5_small_easy_f=0.6', 'CNMC_5_small_easy_f=0.7',
        'CNMC_5_small_easy_f=0.8', 'CNMC_5_small_easy_f=0.9', 'CNMC_5_small_easy_f=1.0',
        'CNMC_5_small_hard_f=0.1', 'CNMC_5_small_hard_f=0.2', 'CNMC_5_small_hard_f=0.3', 'CNMC_5_small_hard_f=0.4',
        'CNMC_5_small_hard_f=0.5', 'CNMC_5_small_hard_f=0.6', 'CNMC_5_small_hard_f=0.7',
        'CNMC_5_small_hard_f=0.8', 'CNMC_5_small_hard_f=0.9', 'CNMC_5_small_hard_f=1.0',
        'CNMC_5_small_clusters_f=0.1', 'CNMC_5_small_clusters_f=0.2', 'CNMC_5_small_clusters_f=0.3', 'CNMC_5_small_clusters_f=0.4',
        'CNMC_5_small_clusters_f=0.5', 'CNMC_5_small_clusters_f=0.6', 'CNMC_5_small_clusters_f=0.7',
        'CNMC_5_small_clusters_f=0.8', 'CNMC_5_small_clusters_f=0.9', 'CNMC_5_small_clusters_f=1.0',
        'CNMC_6', 'CNMC_6_image_rot_f=0.1', 'CNMC_6_image_rot_f=0.2',
        'CNMC_6_image_rot_f=0.3', 'CNMC_6_image_rot_f=0.4', 'CNMC_6_image_rot_f=0.5',
        'CNMC_6_image_rot_f=0.6', 'CNMC_6_image_rot_f=0.7', 'CNMC_6_image_rot_f=0.8',
        'CNMC_6_image_rot_f=0.9', 'CNMC_6_image_rot_f=1.0', 'CNMC_6_image_translation_f=0.1',
        'CNMC_6_image_translation_f=0.2', 'CNMC_6_image_translation_f=0.3', 'CNMC_6_image_translation_f=0.4',
        'CNMC_6_image_translation_f=0.5', 'CNMC_6_image_translation_f=0.6', 'CNMC_6_image_translation_f=0.7',
        'CNMC_6_image_translation_f=0.8', 'CNMC_6_image_translation_f=0.9', 'CNMC_6_image_translation_f=1.0',
        'CNMC_6_image_zoom_f=0.1', 'CNMC_6_image_zoom_f=0.2', 'CNMC_6_image_zoom_f=0.3',
        'CNMC_6_image_zoom_f=0.4', 'CNMC_6_image_zoom_f=0.5', 'CNMC_6_image_zoom_f=0.6',
        'CNMC_6_image_zoom_f=0.7', 'CNMC_6_image_zoom_f=0.8', 'CNMC_6_image_zoom_f=0.9',
        'CNMC_6_image_zoom_f=1.0', 'CNMC_6_add_noise_gaussian_f=0.1', 'CNMC_6_add_noise_gaussian_f=0.2',
        'CNMC_6_add_noise_gaussian_f=0.3', 'CNMC_6_add_noise_gaussian_f=0.4', 'CNMC_6_add_noise_gaussian_f=0.5',
        'CNMC_6_add_noise_gaussian_f=0.6', 'CNMC_6_add_noise_gaussian_f=0.7', 'CNMC_6_add_noise_gaussian_f=0.8',
        'CNMC_6_add_noise_gaussian_f=0.9', 'CNMC_6_add_noise_gaussian_f=1.0', 'CNMC_6_add_noise_poisson_f=0.1',
        'CNMC_6_add_noise_poisson_f=0.2', 'CNMC_6_add_noise_poisson_f=0.3', 'CNMC_6_add_noise_poisson_f=0.4',
        'CNMC_6_add_noise_poisson_f=0.5', 'CNMC_6_add_noise_poisson_f=0.6', 'CNMC_6_add_noise_poisson_f=0.7',
        'CNMC_6_add_noise_poisson_f=0.8', 'CNMC_6_add_noise_poisson_f=0.9', 'CNMC_6_add_noise_poisson_f=1.0',
        'CNMC_6_add_noise_salt_and_pepper_f=0.1', 'CNMC_6_add_noise_salt_and_pepper_f=0.2',
        'CNMC_6_add_noise_salt_and_pepper_f=0.3', 'CNMC_6_add_noise_salt_and_pepper_f=0.4',
        'CNMC_6_add_noise_salt_and_pepper_f=0.5', 'CNMC_6_add_noise_salt_and_pepper_f=0.6',
        'CNMC_6_add_noise_salt_and_pepper_f=0.7', 'CNMC_6_add_noise_salt_and_pepper_f=0.8',
        'CNMC_6_add_noise_salt_and_pepper_f=0.9', 'CNMC_6_add_noise_salt_and_pepper_f=1.0',
        'CNMC_6_add_noise_speckle_f=0.1', 'CNMC_6_add_noise_speckle_f=0.2', 'CNMC_6_add_noise_speckle_f=0.3',
        'CNMC_6_add_noise_speckle_f=0.4', 'CNMC_6_add_noise_speckle_f=0.5', 'CNMC_6_add_noise_speckle_f=0.6',
        'CNMC_6_add_noise_speckle_f=0.7', 'CNMC_6_add_noise_speckle_f=0.8', 'CNMC_6_add_noise_speckle_f=0.9',
        'CNMC_6_add_noise_speckle_f=1.0', 'CNMC_6_imbalance_classes_f=0.1', 'CNMC_6_imbalance_classes_f=0.2',
        'CNMC_6_imbalance_classes_f=0.3', 'CNMC_6_imbalance_classes_f=0.4', 'CNMC_6_imbalance_classes_f=0.5',
        'CNMC_6_imbalance_classes_f=0.6', 'CNMC_6_imbalance_classes_f=0.7', 'CNMC_6_imbalance_classes_f=0.8',
        'CNMC_6_imbalance_classes_f=0.9', 'CNMC_6_imbalance_classes_f=1.0', 'CNMC_6_grayscale_f=0.1',
        'CNMC_6_grayscale_f=0.2', 'CNMC_6_grayscale_f=0.3', 'CNMC_6_grayscale_f=0.4',
        'CNMC_6_grayscale_f=0.5', 'CNMC_6_grayscale_f=0.6', 'CNMC_6_grayscale_f=0.7',
        'CNMC_6_grayscale_f=0.8', 'CNMC_6_grayscale_f=0.9', 'CNMC_6_grayscale_f=1.0',
        'CNMC_6_hsv_f=0.1', 'CNMC_6_hsv_f=0.2', 'CNMC_6_hsv_f=0.3', 'CNMC_6_hsv_f=0.4',
        'CNMC_6_hsv_f=0.5', 'CNMC_6_hsv_f=0.6', 'CNMC_6_hsv_f=0.7',
        'CNMC_6_hsv_f=0.8', 'CNMC_6_hsv_f=0.9', 'CNMC_6_hsv_f=1.0',
        'CNMC_6_blur_f=0.1', 'CNMC_6_blur_f=0.2', 'CNMC_6_blur_f=0.3', 'CNMC_6_blur_f=0.4',
        'CNMC_6_blur_f=0.5', 'CNMC_6_blur_f=0.6', 'CNMC_6_blur_f=0.7',
        'CNMC_6_blur_f=0.8', 'CNMC_6_blur_f=0.9', 'CNMC_6_blur_f=1.0',
        'CNMC_6_small_random_f=0.1', 'CNMC_6_small_random_f=0.2', 'CNMC_6_small_random_f=0.3', 'CNMC_6_small_random_f=0.4',
        'CNMC_6_small_random_f=0.5', 'CNMC_6_small_random_f=0.6', 'CNMC_6_small_random_f=0.7',
        'CNMC_6_small_random_f=0.8', 'CNMC_6_small_random_f=0.9', 'CNMC_6_small_random_f=1.0',
        'CNMC_6_small_easy_f=0.1', 'CNMC_6_small_easy_f=0.2', 'CNMC_6_small_easy_f=0.3', 'CNMC_6_small_easy_f=0.4',
        'CNMC_6_small_easy_f=0.5', 'CNMC_6_small_easy_f=0.6', 'CNMC_6_small_easy_f=0.7',
        'CNMC_6_small_easy_f=0.8', 'CNMC_6_small_easy_f=0.9', 'CNMC_6_small_easy_f=1.0',
        'CNMC_6_small_hard_f=0.1', 'CNMC_6_small_hard_f=0.2', 'CNMC_6_small_hard_f=0.3', 'CNMC_6_small_hard_f=0.4',
        'CNMC_6_small_hard_f=0.5', 'CNMC_6_small_hard_f=0.6', 'CNMC_6_small_hard_f=0.7',
        'CNMC_6_small_hard_f=0.8', 'CNMC_6_small_hard_f=0.9', 'CNMC_6_small_hard_f=1.0',
        'CNMC_6_small_clusters_f=0.1', 'CNMC_6_small_clusters_f=0.2', 'CNMC_6_small_clusters_f=0.3', 'CNMC_6_small_clusters_f=0.4',
        'CNMC_6_small_clusters_f=0.5', 'CNMC_6_small_clusters_f=0.6', 'CNMC_6_small_clusters_f=0.7',
        'CNMC_6_small_clusters_f=0.8', 'CNMC_6_small_clusters_f=0.9', 'CNMC_6_small_clusters_f=1.0']

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

    # output path
    output_path = os.path.join(parent_path, 'outputs/{}'.format(data[dataset]['output_path']))
    data[dataset]['output_path'] = output_path

# create json configuration file
with open('config.json', 'w') as f:
    json.dump(data, f)
