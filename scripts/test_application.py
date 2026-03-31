"""Test script to validate VIVE Labeler functionality.

This script performs automated tests on the VIVE Labeler core modules
using the synthetic test dataset.

Usage:
    python scripts/test_application.py --dataset ./test_dataset
"""

import argparse
import sys
from pathlib import Path

# Import from installed package
try:
    # Try importing from installed package
    from vive_labeler.core.data_loader import LeRobotDataLoader
    from vive_labeler.core.dataset import ViVEDataset
    from vive_labeler.core.synchronizer import DataSynchronizer
    from vive_labeler.core.transforms import Transform3D, quaternion_to_euler
    from vive_labeler.labeling.label_manager import LabelManager
    from vive_labeler.labeling.export import AnnotationExporter
except ImportError:
    # Fallback to adding src to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.core.data_loader import LeRobotDataLoader
    from src.core.dataset import ViVEDataset
    from src.core.synchronizer import DataSynchronizer
    from src.core.transforms import Transform3D, quaternion_to_euler
    from src.labeling.label_manager import LabelManager
    from src.labeling.export import AnnotationExporter


def test_data_loader(dataset_path):
    """Test data loader functionality."""
    print("\n" + "=" * 60)
    print("TEST 1: Data Loader")
    print("=" * 60)

    try:
        loader = LeRobotDataLoader(dataset_path)
        print(f"✅ Dataset loaded successfully")
        print(f"   Episodes: {loader.num_episodes}")
        print(f"   Total samples: {len(loader)}")
        print(f"   Camera keys: {loader.get_camera_keys()}")
        print(f"   State keys: {loader.get_state_keys()}")

        # Test episode loading
        if loader.num_episodes > 0:
            episode = loader.get_episode(0)
            print(f"✅ Episode 0 loaded: {len(episode['samples'])} samples")

        return True
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset(dataset_path):
    """Test ViVEDataset functionality."""
    print("\n" + "=" * 60)
    print("TEST 2: ViVEDataset")
    print("=" * 60)

    try:
        dataset = ViVEDataset(dataset_path, episode_index=0)
        print(f"✅ ViVEDataset loaded")
        print(f"   Episode: {dataset.episode_metadata.episode_id}")
        print(f"   Frames: {dataset.get_frame_count()}")
        print(f"   Duration: {dataset.get_duration():.2f}s")
        print(f"   FPS: {dataset.episode_metadata.fps}")

        # Test frame retrieval
        frame_data = dataset.get_frame(0)
        print(f"✅ Frame 0 loaded")
        print(f"   Video frame shape: {frame_data['frame'].shape}")
        print(f"   Timestamp: {frame_data['timestamp']:.3f}s")
        print(f"   Sensor keys: {list(frame_data['sensor_data'].keys())}")

        # Test VIVE transform
        transform = dataset.get_vive_transform(0)
        if transform:
            print(f"✅ VIVE transform extracted")
            print(f"   Position: {transform.position}")
            print(f"   Quaternion: {transform.quaternion}")
            euler = transform.euler
            print(f"   Euler (rad): [{euler[0]:.3f}, {euler[1]:.3f}, {euler[2]:.3f}]")

        # Test trajectory
        trajectory = dataset.get_trajectory(0, min(50, dataset.get_frame_count()))
        print(f"✅ Trajectory computed: {len(trajectory)} points")

        return True
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_synchronization(dataset_path):
    """Test synchronization functionality."""
    print("\n" + "=" * 60)
    print("TEST 3: Synchronization")
    print("=" * 60)

    try:
        dataset = ViVEDataset(dataset_path, episode_index=0)

        # Get a synchronizer
        camera_key = dataset.episode_metadata.camera_keys[0]
        synchronizer = dataset.synchronizers[camera_key]

        print(f"✅ Synchronizer initialized")
        sync_report = synchronizer.get_sync_report()
        print(f"   Video: {sync_report['video_frame_count']} frames @ {sync_report['video_fps']} fps")
        print(f"   Sensor: {sync_report['sensor_sample_count']} samples @ {1/sync_report['sensor_mean_dt']:.1f} Hz")
        print(f"   Coverage: {'full' if sync_report['has_full_coverage'] else 'partial'}")

        # Test synchronized frame
        synced_frame = synchronizer.get_synced_frame(10)
        print(f"✅ Synchronized frame retrieved")
        print(f"   Frame index: {synced_frame.frame_index}")
        print(f"   Timestamp: {synced_frame.timestamp:.3f}s")
        print(f"   Time diff: {synced_frame.metadata['time_diff_ms']:.2f}ms")

        return True
    except Exception as e:
        print(f"❌ Synchronization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_labeling(dataset_path):
    """Test labeling and export functionality."""
    print("\n" + "=" * 60)
    print("TEST 4: Labeling & Export")
    print("=" * 60)

    try:
        # Create label manager
        label_manager = LabelManager()

        # Add labels
        label1 = label_manager.add_label("test_action_1", color="#FF0000")
        label2 = label_manager.add_label("test_action_2", color="#00FF00")
        print(f"✅ Created {len(label_manager.labels)} labels")

        # Add annotations
        label_manager.add_frame_annotation(10, label1.id)
        label_manager.add_frame_annotation(20, label2.id)
        label_manager.add_interval_annotation(30, 50, label1.id)
        print(f"✅ Created {len(label_manager.annotations)} annotations")

        # Test statistics
        stats = label_manager.get_statistics()
        print(f"   Frame annotations: {stats['frame_annotations']}")
        print(f"   Interval annotations: {stats['interval_annotations']}")

        # Test export
        exporter = AnnotationExporter(label_manager)
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # JSON export
            exporter.export_to_json(tmpdir / "test.json")
            print(f"✅ JSON export successful")

            # CSV export
            exporter.export_to_csv(tmpdir / "test.csv", format_type="frame_based")
            print(f"✅ CSV export successful")

        return True
    except Exception as e:
        print(f"❌ Labeling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transforms():
    """Test 3D transform functionality."""
    print("\n" + "=" * 60)
    print("TEST 5: 3D Transforms")
    print("=" * 60)

    try:
        import numpy as np

        # Test identity transform
        t = Transform3D(position=np.array([1.0, 2.0, 3.0]))
        print(f"✅ Transform created")
        print(f"   Position: {t.position}")
        print(f"   Quaternion: {t.quaternion}")

        # Test point transformation
        point = np.array([0.0, 0.0, 0.0])
        transformed = t.transform_point(point)
        print(f"✅ Point transformation")
        print(f"   Original: {point}")
        print(f"   Transformed: {transformed}")

        # Test inverse
        t_inv = t.inverse()
        print(f"✅ Inverse transform computed")

        # Test composition
        t2 = Transform3D(position=np.array([1.0, 0.0, 0.0]))
        t_composed = t * t2
        print(f"✅ Transform composition")
        print(f"   Composed position: {t_composed.position}")

        return True
    except Exception as e:
        print(f"❌ Transform test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test VIVE Labeler functionality")
    parser.add_argument(
        "--dataset",
        type=str,
        default="./test_dataset",
        help="Path to test dataset (default: ./test_dataset)"
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("\nCreate test dataset first:")
        print("  python scripts/create_test_dataset.py")
        return 1

    print("\n" + "=" * 60)
    print("VIVE LABELER - AUTOMATED TESTS")
    print("=" * 60)
    print(f"Dataset: {dataset_path.absolute()}")

    results = []

    # Run tests
    results.append(("Data Loader", test_data_loader(dataset_path)))
    results.append(("ViVEDataset", test_dataset(dataset_path)))
    results.append(("Synchronization", test_synchronization(dataset_path)))
    results.append(("Labeling & Export", test_labeling(dataset_path)))
    results.append(("3D Transforms", test_transforms()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {name}")

    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)

    if passed == total:
        print("\n🎉 All tests passed! VIVE Labeler is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
