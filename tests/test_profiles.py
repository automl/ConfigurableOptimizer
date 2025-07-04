import unittest

from confopt.profile import (
    BaseProfile,
    DARTSProfile,
    DRNASProfile,
    GDASProfile,
    SNASProfile,
)


class TestBaseProfile(unittest.TestCase):
    def test_config_change(self) -> None:
        profile = BaseProfile(
            "darts",
            epochs=1,
            is_partial_connection=True,
            dropout=0.5,
            perturbation="random",
            perturbator_sample_frequency="step",
            sampler_arch_combine_fn="default",
            entangle_op_weights=False,
            lora_rank=0,
            lora_warm_epochs=0,
            lora_toggle_epochs=None,
            lora_toggle_probability=None,
            seed=100,
            trainer_preset="nb201",
            oles=False,
            calc_gm_score=False,
            prune_epochs=None,
            prune_fractions=None,
            is_arch_attention_enabled=False,
            regularization_config=None,
            pt_select_architecture=False,
        )

        partial_connector_config = {"k": 2}

        perturbator_config = {
            "epsilon": 0.5,
        }

        trainer_config = {"use_data_parallel": True}

        profile.configure_partial_connector(**partial_connector_config)
        profile.configure_perturbator(**perturbator_config)
        profile.configure_trainer(**trainer_config)

        assert profile.partial_connector_config["k"] == partial_connector_config["k"] # type: ignore
        assert profile.perturb_config["epsilon"] == perturbator_config["epsilon"] # type: ignore
        assert (
            profile.trainer_config["use_data_parallel"]
            == trainer_config["use_data_parallel"]
        )

    def test_invalid_configuration(self) -> None:
        profile = BaseProfile(
            "gdas",
            trainer_preset="nb201",
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
        profile = DARTSProfile(
            epochs=100,
            is_partial_connection=True,
            perturbation="random",
            sampler_sample_frequency="step",
            partial_connector_config=partial_connector_config,
            perturbator_config=perturb_config,
            dropout=None,
            sampler_arch_combine_fn="default",
            entangle_op_weights=False,
            lora_rank=0,
            lora_warm_epochs=0,
            lora_toggle_epochs=None,
            lora_toggle_probability=None,
            seed=100,
            trainer_preset="nb201",
            oles=False,
            calc_gm_score=False,
            prune_epochs=None,
            prune_fractions=None,
            is_arch_attention_enabled=False,
            regularization_config=None,
            pt_select_architecture=False,
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
            profile = DARTSProfile(  # noqa: F841
                epochs=100,
                trainer_preset="nb201",
                is_partial_connection=True,
                perturbation="random",
                sampler_sample_frequency="step",
                partial_connector_config=partial_connector_config,
                perturbator_config=perturb_config,
            )

    def test_sampler_change(self) -> None:
        profile = DARTSProfile(
            epochs=100,
            trainer_preset="nb201",
            sampler_sample_frequency="step",
        )
        sampler_config = {"sample_frequency": "epoch"}
        profile.configure_sampler(**sampler_config)
        assert (
            profile.sampler_config["sample_frequency"] # type: ignore
            == sampler_config["sample_frequency"]
        )

        with self.assertRaises(AssertionError):
            profile.configure_sampler(invalid_config="step")

    def test_sampler_post_fn(self) -> None:
        profile = DARTSProfile(epochs=1, trainer_preset="nb201")
        assert profile.sampler_config["arch_combine_fn"] == "default" # type: ignore
        sampler_config = {"arch_combine_fn": "sigmoid"}
        profile.configure_sampler(**sampler_config)
        assert (
            profile.sampler_config["arch_combine_fn"] == sampler_config["arch_combine_fn"] # type: ignore
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
            dropout=None,
            sampler_arch_combine_fn="default",
            entangle_op_weights=False,
            lora_rank=0,
            lora_warm_epochs=0,
            lora_toggle_epochs=None,
            lora_toggle_probability=None,
            seed=100,
            trainer_preset="nb201",
            oles=False,
            calc_gm_score=False,
            prune_epochs=None,
            prune_fractions=None,
            is_arch_attention_enabled=False,
            regularization_config=None,
            pt_select_architecture=False,
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
                trainer_preset="nb201",
                is_partial_connection=True,
                perturbation="random",
                sampler_sample_frequency="step",
                partial_connector_config=partial_connector_config,
                perturbator_config=perturb_config,
            )

    def test_sampler_change(self) -> None:
        profile = DRNASProfile(
            epochs=100,
            trainer_preset="nb201",
            sampler_sample_frequency="step",
        )
        sampler_config = {"sample_frequency": "epoch"}
        profile.configure_sampler(**sampler_config)
        assert (
            profile.sampler_config["sample_frequency"] # type: ignore
            == sampler_config["sample_frequency"]
        )

        with self.assertRaises(AssertionError):
            profile.configure_sampler(invalid_config="step")

    def test_sampler_post_fn(self) -> None:
        profile = DRNASProfile(epochs=1, trainer_preset="nb201",)
        assert profile.sampler_config["arch_combine_fn"] == "default" # type: ignore
        sampler_config = {"arch_combine_fn": "sigmoid"}
        profile.configure_sampler(**sampler_config)
        assert (
            profile.sampler_config["arch_combine_fn"] == sampler_config["arch_combine_fn"] # type: ignore
        )


class TestGDASProfile(unittest.TestCase):
    def test_initialization(self) -> None:
        perturb_config = {"epsilon": 0.5}
        partial_connector_config = {
            "k": 2,
        }
        profile = GDASProfile(
            epochs=100,
            tau_min=0.5,
            tau_max=50,
            is_partial_connection=True,
            perturbation="random",
            sampler_sample_frequency="step",
            partial_connector_config=partial_connector_config,
            perturbator_config=perturb_config,
            dropout=None,
            sampler_arch_combine_fn="default",
            entangle_op_weights=False,
            lora_rank=0,
            lora_warm_epochs=0,
            lora_toggle_epochs=None,
            lora_toggle_probability=None,
            seed=100,
            trainer_preset="nb201",
            oles=False,
            calc_gm_score=False,
            prune_epochs=None,
            prune_fractions=None,
            is_arch_attention_enabled=False,
            regularization_config=None,
            pt_select_architecture=False,
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
                trainer_preset="nb201",
                is_partial_connection=True,
                perturbation="random",
                sampler_sample_frequency="step",
                partial_connector_config=partial_connector_config,
                perturbator_config=perturb_config,
            )

    def test_sampler_change(self) -> None:
        profile = GDASProfile(
            epochs=100,
            trainer_preset="nb201",
            sampler_sample_frequency="step",
        )
        sampler_config = {"tau_max": 12, "tau_min": 0.3}
        profile.configure_sampler(**sampler_config)

        assert profile.sampler_config["tau_max"] == sampler_config["tau_max"] # type: ignore
        assert profile.sampler_config["tau_min"] == sampler_config["tau_min"] # type: ignore

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
            temp_init=1.0,
            temp_min=0.33,
            temp_annealing=True,
            is_partial_connection=True,
            perturbation="random",
            sampler_sample_frequency="step",
            partial_connector_config=partial_connector_config,
            perturbator_config=perturb_config,
            dropout=None,
            sampler_arch_combine_fn="default",
            entangle_op_weights=False,
            lora_rank=0,
            lora_warm_epochs=0,
            lora_toggle_epochs=None,
            lora_toggle_probability=None,
            seed=100,
            trainer_preset="nb201",
            oles=False,
            calc_gm_score=False,
            prune_epochs=None,
            prune_fractions=None,
            is_arch_attention_enabled=False,
            regularization_config=None,
            pt_select_architecture=False,
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
                trainer_preset="nb201",
                is_partial_connection=True,
                perturbation="random",
                sampler_sample_frequency="step",
                partial_connector_config=partial_connector_config,
                perturbator_config=perturb_config,
            )

    def test_sampler_change(self) -> None:
        profile = SNASProfile(
            epochs=100,
            trainer_preset="nb201",
            sampler_sample_frequency="step",
        )
        sampler_config = {"temp_min": 0.5, "temp_init": 1.3}
        profile.configure_sampler(**sampler_config)
        assert profile.sampler_config["temp_min"] == sampler_config["temp_min"] # type: ignore
        assert profile.sampler_config["temp_init"] == sampler_config["temp_init"] # type: ignore

        with self.assertRaises(AssertionError):
            profile.configure_sampler(invalid_config="step")


if __name__ == "__main__":
    unittest.main()
