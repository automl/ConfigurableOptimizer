import unittest

from confopt.profiles import (
    DartsProfile,
    DRNASProfile,
    GDASProfile,
    ProfileConfig,
    SNASProfile,
)


class TestProfileConfig(unittest.TestCase):
    def test_config_change(self) -> None:
        profile = ProfileConfig(
            "TEST",
            epochs=1,
            is_partial_connection=True,
            dropout=0.5,
            perturbation="random",
            perturbator_sample_frequency="step",
        )

        partial_connector_config = {"k": 2}

        perturbator_config = {
            "epsilon": 0.5,
        }

        trainer_config = {"use_data_parallel": True}

        profile.configure_partial_connector(**partial_connector_config)
        profile.configure_perturbator(**perturbator_config)
        profile.configure_trainer(**trainer_config)

        assert profile.partial_connector_config["k"] == partial_connector_config["k"]
        assert profile.perturb_config["epsilon"] == perturbator_config["epsilon"]
        assert (
            profile.trainer_config["use_data_parallel"]
            == trainer_config["use_data_parallel"]
        )

    def test_invalid_configuration(self) -> None:
        profile = ProfileConfig(
            "TEST",
            epochs=1,
            is_partial_connection=True,
            dropout=0.5,
            perturbation="random",
            perturbator_sample_frequency="step",
        )

        partial_connector_config = {"invalid_config": 2}

        perturbator_config = {
            "invalid_config": 0.5,
        }

        trainer_config = {"invalid_config": False}

        dropout_config = {"invalid_config": "test"}

        with self.assertRaises(AssertionError):
            profile.configure_partial_connector(**partial_connector_config)

        with self.assertRaises(AssertionError):
            profile.configure_perturbator(**perturbator_config)

        with self.assertRaises(AssertionError):
            profile.configure_trainer(**trainer_config)

        with self.assertRaises(AssertionError):
            profile.configure_dropout(**dropout_config)


class TestDartsProfile(unittest.TestCase):
    def test_initialization(self) -> None:
        perturb_config = {"epsilon": 0.5}
        partial_connector_config = {
            "k": 2,
        }
        profile = DartsProfile(
            epochs=100,
            is_partial_connection=True,
            perturbation="random",
            sampler_sample_frequency="step",
            partial_connector_config=partial_connector_config,
            perturbator_config=perturb_config,
        )

        assert profile.sampler_config is not None
        assert profile.partial_connector_config["k"] == partial_connector_config["k"]
        assert profile.perturb_config["epsilon"] == perturb_config["epsilon"]

    def test_invalid_initialization(self) -> None:
        perturb_config = {"invalid_config": 0.5}
        partial_connector_config = {
            "invalid_config": 2,
        }

        with self.assertRaises(AssertionError):
            profile = DartsProfile(  # noqa: F841
                epochs=100,
                is_partial_connection=True,
                perturbation="random",
                sampler_sample_frequency="step",
                partial_connector_config=partial_connector_config,
                perturbator_config=perturb_config,
            )

    def test_sampler_change(self) -> None:
        profile = DartsProfile(
            epochs=100,
            sampler_sample_frequency="step",
        )
        sampler_config = {"sample_frequency": "epoch"}
        profile.configure_sampler(**sampler_config)
        assert (
            profile.sampler_config["sample_frequency"]
            == sampler_config["sample_frequency"]
        )

        with self.assertRaises(AssertionError):
            profile.configure_sampler(invalid_config="step")

    def test_sampler_post_fn(self) -> None:
        profile = DartsProfile(epochs=1)
        assert profile.sampler_config["post_sample_fn"] == "default"
        sampler_config = {"post_sample_fn": "sigmoid"}
        profile.configure_sampler(**sampler_config)
        assert (
            profile.sampler_config["post_sample_fn"] == sampler_config["post_sample_fn"]
        )


class TestDRNASProfile(unittest.TestCase):
    def test_initialization(self) -> None:
        perturb_config = {"epsilon": 0.5}
        partial_connector_config = {
            "k": 2,
        }
        profile = DRNASProfile(
            epochs=100,
            is_partial_connection=True,
            perturbation="random",
            sampler_sample_frequency="step",
            partial_connector_config=partial_connector_config,
            perturbator_config=perturb_config,
        )

        assert profile.sampler_config is not None
        assert profile.partial_connector_config["k"] == partial_connector_config["k"]
        assert profile.perturb_config["epsilon"] == perturb_config["epsilon"]

    def test_invalid_initialization(self) -> None:
        perturb_config = {"invalid_config": 0.5}
        partial_connector_config = {
            "invalid_config": 2,
        }

        with self.assertRaises(AssertionError):
            profile = DRNASProfile(  # noqa: F841
                epochs=100,
                is_partial_connection=True,
                perturbation="random",
                sampler_sample_frequency="step",
                partial_connector_config=partial_connector_config,
                perturbator_config=perturb_config,
            )

    def test_sampler_change(self) -> None:
        profile = DRNASProfile(
            epochs=100,
            sampler_sample_frequency="step",
        )
        sampler_config = {"sample_frequency": "epoch"}
        profile.configure_sampler(**sampler_config)
        assert (
            profile.sampler_config["sample_frequency"]
            == sampler_config["sample_frequency"]
        )

        with self.assertRaises(AssertionError):
            profile.configure_sampler(invalid_config="step")

    def test_sampler_post_fn(self) -> None:
        profile = DRNASProfile(epochs=1)
        assert profile.sampler_config["post_sample_fn"] == "default"
        sampler_config = {"post_sample_fn": "sigmoid"}
        profile.configure_sampler(**sampler_config)
        assert (
            profile.sampler_config["post_sample_fn"] == sampler_config["post_sample_fn"]
        )


class TestGDASProfile(unittest.TestCase):
    def test_initialization(self) -> None:
        perturb_config = {"epsilon": 0.5}
        partial_connector_config = {
            "k": 2,
        }
        profile = GDASProfile(
            epochs=100,
            is_partial_connection=True,
            perturbation="random",
            sampler_sample_frequency="step",
            partial_connector_config=partial_connector_config,
            perturbator_config=perturb_config,
        )

        assert profile.sampler_config is not None
        assert profile.partial_connector_config["k"] == partial_connector_config["k"]
        assert profile.perturb_config["epsilon"] == perturb_config["epsilon"]

    def test_invalid_initialization(self) -> None:
        perturb_config = {"invalid_config": 0.5}
        partial_connector_config = {
            "invalid_config": 2,
        }

        with self.assertRaises(AssertionError):
            profile = GDASProfile(  # noqa: F841
                epochs=100,
                is_partial_connection=True,
                perturbation="random",
                sampler_sample_frequency="step",
                partial_connector_config=partial_connector_config,
                perturbator_config=perturb_config,
            )

    def test_sampler_change(self) -> None:
        profile = GDASProfile(
            epochs=100,
            sampler_sample_frequency="step",
        )
        sampler_config = {"tau_max": 12, "tau_min": 0.3}
        profile.configure_sampler(**sampler_config)

        assert profile.sampler_config["tau_max"] == sampler_config["tau_max"]
        assert profile.sampler_config["tau_min"] == sampler_config["tau_min"]

        with self.assertRaises(AssertionError):
            profile.configure_sampler(invalid_config="step")


class TestSNASProfile(unittest.TestCase):
    def test_initialization(self) -> None:
        perturb_config = {"epsilon": 0.5}
        partial_connector_config = {
            "k": 2,
        }
        profile = SNASProfile(
            epochs=100,
            is_partial_connection=True,
            perturbation="random",
            sampler_sample_frequency="step",
            partial_connector_config=partial_connector_config,
            perturbator_config=perturb_config,
        )

        assert profile.sampler_config is not None
        assert profile.partial_connector_config["k"] == partial_connector_config["k"]
        assert profile.perturb_config["epsilon"] == perturb_config["epsilon"]

    def test_invalid_initialization(self) -> None:
        perturb_config = {"invalid_config": 0.5}
        partial_connector_config = {
            "invalid_config": 2,
        }

        with self.assertRaises(AssertionError):
            profile = SNASProfile(  # noqa: F841
                epochs=100,
                is_partial_connection=True,
                perturbation="random",
                sampler_sample_frequency="step",
                partial_connector_config=partial_connector_config,
                perturbator_config=perturb_config,
            )

    def test_sampler_change(self) -> None:
        profile = SNASProfile(
            epochs=100,
            sampler_sample_frequency="step",
        )
        sampler_config = {"temp_min": 0.5, "temp_init": 1.3}
        profile.configure_sampler(**sampler_config)
        assert profile.sampler_config["temp_min"] == sampler_config["temp_min"]
        assert profile.sampler_config["temp_init"] == sampler_config["temp_init"]

        with self.assertRaises(AssertionError):
            profile.configure_sampler(invalid_config="step")


if __name__ == "__main__":
    unittest.main()
