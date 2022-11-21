import glob
import os
import numpy as np
import cv2
import logging

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

FORCE = False
DISPLAY_SIZE = (1750 // 4, 3100 // 4)
BACK_KEY = 52
FORWARD_KEY = 54
SKIP_KEY = 8  # i.e. backspace

KEY_GROUP_MAP = {49: 1, 50: 2, 51: 3}
PALLET = {0: (255, 255, 0),
          1: (255, 0, 0),
          2: (0, 255, 0),
          3: (0, 0, 255)}
ENTER_KEY = 13
ESC_KEY = 27
MIN_CONTOUR_AREA = 100

def display_results(image, groups, contours, final = False):
    im_cp = np.copy(image)
    for g, c in zip(groups, contours):
        cv2.drawContours(im_cp, [c], 0, PALLET[g], -1)
        display_out = cv2.resize(im_cp, DISPLAY_SIZE)
    im_cp = cv2.resize(im_cp, DISPLAY_SIZE)
    cv2.imshow("result", im_cp)
    if final:
        logging.warning("Validate image with ENTER or restart with ESC")
        while True:
            k = cv2.waitKey(-1)
            if k == ENTER_KEY:
                return True
            elif k == ESC_KEY:
                return False
            else:
                logging.warning(f"Unexpected key: {k}")
    else:
        cv2.waitKey(1)




if __name__ == '__main__':

    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    for f in sorted(glob.glob("data/*.jpg")):
        im = cv2.imread(f)
        mask_file = os.path.splitext(f)[0] + ".png"
        skipped_file = os.path.splitext(f)[0] + "-skipped.txt"
        output_file = os.path.splitext(f)[0] + "-instances.png"
        if (os.path.exists(output_file) or os.path.exists(skipped_file)) and not FORCE:
            logging.info(f"Skipping {os.path.basename(output_file)}")
            continue


        mask = 255 - cv2.cvtColor(cv2.imread(mask_file), cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
        # fixme filter mini contours here!

        validated = False
        skipped = False
        while not validated and not skipped:
            logging.info(f"Annotating {os.path.basename(f)}")

            i = 0
            groups = [0] * len(contours)
            display_results(im, groups, contours)
            while not skipped and np.any(np.array(groups) == 0):

                c = contours[i]
                canvas = np.copy(im)
                current_group = groups[i]
                cv2.drawContours(canvas, [c], 0, PALLET[current_group], -1)
                canvas = cv2.resize(canvas, DISPLAY_SIZE)
                cv2.imshow("window", canvas)
                k = cv2.waitKey(-1)
                logging.info(f"Pressed {k}")
                if k in KEY_GROUP_MAP.keys():
                    groups[i] = KEY_GROUP_MAP[k]
                    logging.info(f"Allocated group {groups[i]}")
                    i += 1
                elif k == SKIP_KEY:
                    logging.warning(f"Skipping this {f}")
                    skipped = True

                elif k == BACK_KEY:
                    logging.warning(f"Going back")
                    i -= 1
                elif k == FORWARD_KEY:
                    logging.warning(f"Going forward")
                    i += 1
                else:
                    logging.warning(f"Unexpected key: {k}")

                if i < 0:
                    i = 0
                if i >= len(contours):
                    i = len(contours) - 1
                if not skipped:
                    display_results(im, groups, contours)
            if not skipped:
                validated = display_results(im, groups, contours, final=True)

        if skipped:
            logging.warning(f"Creating skipped stamp {skipped_file}")
            with open(skipped_file, 'w') as f:
                f.write("")
            continue
        logging.info(f"Done with {os.path.basename(f)}. Saving {output_file}")
        output = np.zeros_like(im)

        # this is where we map each instance as a different colour
        for g, c in zip(groups, contours):
            cv2.drawContours(output, [c], 0, PALLET[g], -1)
        cv2.imwrite(output_file, output)

        # alternatively, we can just save multiple files:

        for g, c in zip(groups, contours):
            output = np.zeros_like(im)
            output_instance_file = os.path.splitext(f)[0] + f"-instance_{g}.png"
            cv2.drawContours(output, [c], 0, (255,255,255), -1)
            cv2.imwrite(output_instance_file, output)
