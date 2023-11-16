import unittest

from mix_n_match.correlations import pair_data


class TestCorrelations(unittest.TestCase):
    def test_correlations(self):
        pass

    def test_pair_data(self):
        list_of_items = [1, 2]
        # -- test default: method='full' and ignore_diagonal=False
        output_list = list(pair_data(iter(list_of_items)))
        expected_list = [
            ((0, 0), 1, 1),
            ((0, 1), 1, 2),
            ((1, 0), 2, 1),
            ((1, 1), 2, 2),
        ]
        assert output_list == expected_list

        # -- test upper
        output_list = list(pair_data(iter(list_of_items), method="upper"))
        expected_list = [((0, 0), 1, 1), ((0, 1), 1, 2), ((1, 1), 2, 2)]
        assert output_list == expected_list

        # -- test lower
        output_list = list(pair_data(iter(list_of_items), method="lower"))
        expected_list = [((0, 0), 1, 1), ((1, 0), 2, 1), ((1, 1), 2, 2)]
        assert output_list == expected_list

        # -- test ignore diagonal
        output_list = list(
            pair_data(iter(list_of_items), ignore_diagonal=True)
        )
        expected_list = [
            ((0, 1), 1, 2),
            ((1, 0), 2, 1),
        ]
        assert output_list == expected_list

        # -- test upper
        output_list = list(
            pair_data(
                iter(list_of_items), method="upper", ignore_diagonal=True
            )
        )
        expected_list = [((0, 1), 1, 2)]
        assert output_list == expected_list

        # -- test lower
        output_list = list(
            pair_data(
                iter(list_of_items), method="lower", ignore_diagonal=True
            )
        )
        expected_list = [((1, 0), 2, 1)]
        assert output_list == expected_list


if __name__ == "__main__":
    unittest.main()
