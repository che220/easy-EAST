import cv2


def show_image_in_window(img, win_name, win_size, win_location=(0, 0), should_wait=False):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, win_size[0], win_size[1])
    cv2.moveWindow(win_name, win_location[0], win_location[1])
    cv2.imshow(win_name, img)
    if should_wait:
        cv2.waitKey(0) & 0xFF  # for 64-bit machine
        cv2.destroyAllWindows()
