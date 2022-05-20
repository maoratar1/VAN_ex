# == Helpers for debugging == #
def check1(pair0, left0_left1_matches):
    """
    check if the matches in pair0 and left0_left1_matches are the same
    """
    n = len(pair0)
    for i in range(n):
        p0_idx = pair0[i].queryIdx
        l0_l1_idx = left0_left1_matches[i].queryIdx

        if p0_idx != l0_l1_idx:
            print(f"Wrong {i}")


def check2(pair1, left0_left1_matches):
    """
    check if the matches in pair1 and left0_left1_matches are the same
    """
    n = len(pair1)
    for i in range(n):
        p1_idx = pair1[i].queryIdx
        l0_l1_idx = left0_left1_matches[i].trainIdx

        if p1_idx != l0_l1_idx:
            print(f"Wrong {i}")