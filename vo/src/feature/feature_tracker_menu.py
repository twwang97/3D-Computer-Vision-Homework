
from vo.src.feature.feature_tracker_configs import FeatureTrackerConfigs
from vo.src.feature.feature_tracker import feature_tracker_factory

def feature_tracker_selector(config_name, num_features):

  # select your tracker configuration (see the file feature_tracker_configs.py) 
  # LK_SHI_TOMASI, LK_FAST
  # SHI_TOMASI_ORB, FAST_ORB, ORB, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, SUPERPOINT, FAST_TFEAT

  if config_name == 'orb':
    tracker_config = FeatureTrackerConfigs.ORB
  elif config_name == 'LK_SHI_TOMASI':
    tracker_config = FeatureTrackerConfigs.LK_SHI_TOMASI
  elif config_name == 'LK_FAST':
    tracker_config = FeatureTrackerConfigs.LK_FAST
  elif config_name == 'brisk':
    tracker_config = FeatureTrackerConfigs.BRISK
  elif config_name == 'akaze':
    tracker_config = FeatureTrackerConfigs.AKAZE
  elif config_name == 'sift':
    tracker_config = FeatureTrackerConfigs.SIFT
  elif config_name == 'sift_root':
    tracker_config = FeatureTrackerConfigs.ROOT_SIFT
  else:
    print('default tracker: ORB')
    tracker_config = FeatureTrackerConfigs.ORB

  tracker_config['num_features'] = num_features
  feature_tracker = feature_tracker_factory(**tracker_config)

  return feature_tracker