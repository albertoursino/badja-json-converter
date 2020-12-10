from unittest import TestCase

import write


class Test(TestCase):

    def test_replace_slash(self):
        self.assertEqual(write.replace_slash('\\'), '/')

    def test_swap_coordinates(self):
        self.assertEqual(write.swap_coordinates([[1, 0], [1, 0]]), [[0, 1], [0, 1]])

    def test_resize_coordinates(self):
        self.assertEqual(write.resize_coordinates([[10, 10], [10, 0], [0, 0]], 0.5, 0.5), [[5, 5], [5, 0], [0, 0]])
