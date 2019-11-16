"""
@title
@description
"""
import os

import tensorflow_datasets as tfds

from SystemControl import DATA_DIR


class PhysioHeatmapDataset(tfds.core.GeneratorBasedBuilder):
    """
    Dataset of heatmaps of EEG maps for a specified individual in the Physio datasset
    """

    VERSION = tfds.core.Version('0.1.0')

    def __init__(self, datasource_name: str, subject_name: str, window_length_list: list):
        super().__init__()

        self.datasource_name = datasource_name
        self.subject_name = subject_name
        base_data_dir = os.path.join(
            DATA_DIR,
            'heatmaps',
            f'data_source_{datasource_name}'
        )
        self.data_dirs = {}
        for wnd_length in window_length_list:
            window_dir = os.path.join(base_data_dir, f'window_length_{wnd_length}')
            self.data_dirs[wnd_length] = window_dir
        return

    def _info(self):
        ds_info = tfds.core.DatasetInfo(
            builder=self,
            description=("This is the dataset for xxx. It contains yyy. The "
                         "images are kept at their original dimensions."),
            features=tfds.features.FeaturesDict({
                "image_description": tfds.features.Text(),
                "image": tfds.features.Image(),
                # Here, labels can be of 5 distinct values.
                "label": tfds.features.ClassLabel(num_classes=5),
            }),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=("image", "label"),
            homepage="https://dataset-homepage.org",
            # Bibtex citation for the dataset
            citation=
            r"""@article{physio_dataset,
                author = {
                    Goldberger AL,
                    Amaral LAN,
                    Glass L,
                    Hausdorff JM,
                    Ivanov PCh,
                    Mark RG,
                    Mietus JE,
                    Moody GB,
                    Peng C-K"
                }
            """,
        )
        pass  # TODO

    def _split_generators(self, dl_manager):
        # Downloads the data and defines the splits
        # dl_manager is a tfds.download.DownloadManager that can be used to
        # download and extract URLs
        pass  # TODO

    def _generate_examples(self):
        # Yields examples from the dataset
        yield 'key', {}


def main():
    return


if __name__ == '__main__':
    main()
