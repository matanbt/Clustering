from dataclasses import dataclass
import initialization


@dataclass
class ProgramArgs:
    n: int
    k: int
    random: bool


class TestCheckUserInput:
    def test_random_args(self):
        args = ProgramArgs(0, 0, True)
        assert initialization.check_user_input(args)

    def test_zero_n(self):
        args = ProgramArgs(0, 1, False)
        assert not initialization.check_user_input(args)

    def test_negative_n(self):
        args = ProgramArgs(-1, 1, False)
        assert not initialization.check_user_input(args)

    def test_zero_k(self):
        args = ProgramArgs(1, 0, False)
        assert not initialization.check_user_input(args)

    def test_negative_k(self):
        args = ProgramArgs(1, -1, False)
        assert not initialization.check_user_input(args)

    def test_equal_n_k(self):
        args = ProgramArgs(1, 1, False)
        assert not initialization.check_user_input(args)

    def test_smaller_n_k(self):
        args = ProgramArgs(1, 2, False)
        assert not initialization.check_user_input(args)

    def test_bigger_n_k(self):
        args = ProgramArgs(2, 1, False)
        assert initialization.check_user_input(args)


class TestGeneratePoints:
    def test_random_points(self):
        guaranteed_random = False
        args = ProgramArgs(50, 10, True)

        # Probability test of randomness
        for _ in range(10):
            params, points, centers = initialization.generate_points(args)
            if params.n != args.n or params.k != args.k:
                guaranteed_random = True
            assert len(points) == params.n
            assert len(centers) == params.n
            assert params.random
            assert params.dim in [2, 3]

        assert guaranteed_random

    def test_non_random_points(self):
        args = ProgramArgs(50, 10, False)
        params, points, centers = initialization.generate_points(args)

        assert params.n == args.n
        assert params.k == args.k
        assert not params.random
        assert params.dim in [2, 3]
        assert len(points) == args.n
        assert len(centers) == args.n
