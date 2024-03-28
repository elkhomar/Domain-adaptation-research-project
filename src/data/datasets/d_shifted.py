from sklearn.model_selection import train_test_split
from torch.fx.experimental.symbolic_shapes import TrueDiv
from torch.utils.data import Dataset
from skada import source_target_split
from skada.datasets import make_shifted_datasets


class ShiftedDataset(Dataset):

    def __init__(self, source: bool, shift="covariate_shift"):
        self.source = source
        self.shift = shift
        self.all_outputs, self.train_outputs, self.val_outputs = self.load_shifted()

    def load_shifted(self):
        """
        Loads the Shifted gausssians dataset from Skada and returns the features and labels
        """
        X, y, sample_domain = make_shifted_datasets(
            n_samples_source=20,
            n_samples_target=20,
            shift=self.shift,
            noise=0.3,
            label="binary",
            random_state=42,
        )

        X_source, X_target, y_source, y_target = source_target_split(
            X, y, sample_domain=sample_domain
        )

        if self.source:
            all_outputs = (X_source, y_source)
        else:
            all_outputs = (X_target, y_target)

        train_features, val_features, train_labels, val_labels = train_test_split(
            all_outputs[0],
            all_outputs[1],
            test_size=0.2,
            random_state=42,
            stratify=all_outputs[1],
        )
        train_outputs = (train_features, train_labels)
        val_outputs = (val_features, val_labels)
        all_outputs = (all_outputs[0], all_outputs[1])

        return all_outputs, train_outputs, val_outputs
