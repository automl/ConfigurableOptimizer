import os
from pathlib import Path
import shutil
import unittest

import torch

from confopt.utils import (
    Logger,
    prepare_logger,
    get_pos_reductions_darts,
    get_pos_new_cell_darts,
)
from confopt.utils.channel_shuffle import channel_shuffle


class TestUtils(unittest.TestCase):
    def test_prepare_logger(self) -> None:
        save_dir = Path(".") / "tests" / "logs"
        exp_name = "test_exp"
        logger = prepare_logger(save_dir=str(save_dir), seed=9001, exp_name=exp_name)
        checkpoints_path = logger.path(mode="checkpoints")
        assert os.path.exists(checkpoints_path)
        assert os.path.exists(logger.logger_path)

        shutil.rmtree(save_dir, ignore_errors=True)
        logger.close()

    def test_logger(self) -> None:
        save_dir = Path(".") / "tests" / "logs"
        exp_name = "test_exp"
        logger = Logger(
            log_dir=str(save_dir), seed="22", exp_name=exp_name, search_space="nb201"
        )

        checkpoints_path = logger.path(mode="checkpoints")
        assert os.path.exists(checkpoints_path)
        assert os.path.exists(logger.logger_path)

        shutil.rmtree(save_dir, ignore_errors=True)
        logger.close()

    def test_channel_shuffle(self) -> None:
        k = 4
        x = torch.randn(2, 8, 3, 5)
        original_x = x.detach().clone()
        shuffled_x = channel_shuffle(x, groups=k)
        assert x.shape == shuffled_x.shape
        assert not torch.allclose(original_x, shuffled_x)

    def test_get_pos_reductions_darts(self) -> None:
        # Test when layers = 2 nr
        self.assertEqual(get_pos_reductions_darts(2), (1, 2))
        # Test when layers = 3 nrn
        self.assertEqual(get_pos_reductions_darts(3), (1, 3))
        # Test when layers = 4 nrnr
        self.assertEqual(get_pos_reductions_darts(4), (1, 3))
        # Test when layers = 5 nrnrn
        self.assertEqual(get_pos_reductions_darts(5), (1, 3))
        # Test when layers = 6 nnrnrn
        self.assertEqual(get_pos_reductions_darts(6), (2, 4))
        # Test when layers = 7 nnrnnrn
        self.assertEqual(get_pos_reductions_darts(7), (2, 5))
        # Test when layers = 8 nnrnnrnn
        self.assertEqual(get_pos_reductions_darts(8), (2, 5))
        # Test when layers = 9 nnnrnnrnn
        self.assertEqual(get_pos_reductions_darts(9), (3, 6))

    def test_get_pos_new_cell_darts(self) -> None:
        for i in range(2, 4):
            self.assertEqual(get_pos_new_cell_darts(i), i)
        self.assertEqual(get_pos_new_cell_darts(12), 8)
        self.assertEqual(get_pos_new_cell_darts(19), 19)


class TestLogger(unittest.TestCase):
    def test_logger_init_with_runtime(self) -> None:
        log_dir = str(Path(".") / "tests" / "logs")
        logger_source = Logger(
            log_dir=log_dir,
            exp_name="testiiiing",
            search_space="darts",
            dataset="cifar100",
            seed="12",
            use_supernet_checkpoint=False,
        )
        expr_path = "/".join(
            [
                log_dir,
                "testiiiing",
                "darts",
                "cifar100",
                "12",
                "discrete",
            ]
        )
        logger = Logger(
            log_dir=log_dir,
            exp_name="testiiiing",
            search_space="darts",
            dataset="cifar100",
            seed="12",
            runtime=logger_source.runtime,
            use_supernet_checkpoint=False,
        )
        assert logger_source.runtime == logger.runtime
        assert os.path.exists("/".join([expr_path, logger.runtime, "log"]))

    def test_logger_init_with_last_run(self) -> None:
        log_dir = str(Path(".") / "tests" / "logs")
        logger_source = Logger(
            log_dir=log_dir,
            exp_name="testiiiing",
            search_space="darts",
            dataset="cifar100",
            seed="12",
            use_supernet_checkpoint=False,
        )
        expr_path = "/".join(
            [
                log_dir,
                "testiiiing",
                "darts",
                "cifar100",
                "12",
                "discrete",
            ]
        )
        logger = Logger(
            log_dir=log_dir,
            exp_name="testiiiing",
            search_space="darts",
            dataset="cifar100",
            seed="12",
            use_supernet_checkpoint=False,
            last_run=True,
        )
        assert logger_source.runtime == logger.runtime
        assert os.path.exists("/".join([expr_path, logger.runtime, "log"]))

    def test_logger_init(self) -> None:
        log_dir = str(Path(".") / "tests" / "logs")
        expr_path = "/".join(
            [
                log_dir,
                "testiiiing",
                "darts",
                "cifar100",
                "12",
                "supernet",
            ]
        )
        logger = Logger(
            log_dir=log_dir,
            exp_name="testiiiing",
            search_space="darts",
            dataset="cifar100",
            seed="12",
            use_supernet_checkpoint=True,
        )
        assert os.path.exists(expr_path)
        assert os.path.exists("/".join([expr_path, "last_run"]))
        assert os.path.exists("/".join([expr_path, logger.runtime, "log"]))
        assert os.path.exists("/".join([expr_path, logger.runtime, "checkpoints"]))

    # def test_set_up_new_run(self) -> None:
    #     ...

    # def test_set_up_run(self) -> None:
    #     ...

    def test_expr_log_path(self) -> None:
        log_dir = str(Path(".") / "tests" / "logs")
        expr_path = "/".join(
            [
                log_dir,
                "testiiiing",
                "darts",
                "cifar100",
                "12",
                "discrete",
            ]
        )
        logger = Logger(
            log_dir=log_dir,
            exp_name="testiiiing",
            search_space="darts",
            dataset="cifar100",
            seed="12",
            use_supernet_checkpoint=False,
        )
        assert logger.expr_log_path() == Path(expr_path)

    def test_load_last_run(self) -> None:
        log_dir = str(Path(".") / "tests" / "logs")
        "/".join(
            [
                log_dir,
                "testiiiing",
                "darts",
                "cifar100",
                "12",
                "discrete",
            ]
        )
        logger = Logger(
            log_dir=log_dir,
            exp_name="testiiiing",
            search_space="darts",
            dataset="cifar100",
            seed="12",
            use_supernet_checkpoint=False,
        )
        assert logger.load_last_run() == logger.runtime

    # def test_save_last_run(self) -> None:
    #     ...

    def test_path(self) -> None:
        log_dir = str(Path(".") / "tests" / "logs")

        logger = Logger(
            log_dir=log_dir,
            exp_name="testiiiing",
            search_space="darts",
            dataset="cifar100",
            seed="12",
            use_supernet_checkpoint=False,
        )
        expr_path = "/".join(
            [
                log_dir,
                "testiiiing",
                "darts",
                "cifar100",
                "12",
                "discrete",
                logger.runtime,
            ]
        )
        assert logger.path("best_model") == "/".join([expr_path, "best_model.pth"])
        assert logger.path("checkpoints") == "/".join([expr_path, "checkpoints"])
        assert logger.path("log") == "/".join([expr_path, "log"])
        # assert logger.path("last_checkpoint")=='/'.join([expr_path, "best_model.pth"])

    # def test_log(self) -> None:
    #     ...


if __name__ == "__main__":
    unittest.main()
