import numpy as np
import parameter
# Define a class to receive the characteristics of each line detection


class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None

        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = []

        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        self.not_detected_counter = 0

    def check_plausibility(self, lane_fit):
        # check
        return True

    def calc_curvature(self):
        ploty = np.linspace(0, 719, 720)
        y_eval = np.max(ploty)

        # Calc new polynomial coefficients in world space
        if self.best_fit is not None:
            fitA = (parameter.xm_per_pix / parameter.ym_per_pix ** 2) * self.best_fit[0]
            fitB = (parameter.xm_per_pix / parameter.ym_per_pix) * self.best_fit[1]

        # Calculate radius in meter
        self.radius_of_curvature = (1+(2*fitA*y_eval*parameter.ym_per_pix+fitB)**2)**(3/2)/np.absolute(2*fitA)

    def update_best_fit(self, lane_fit):
        if not self.check_plausibility(lane_fit):
            self.not_detected_counter = self.not_detected_counter + 1

        elif lane_fit is not None:
            if self.best_fit is not None:
                self.diffs = abs(lane_fit - self.best_fit)

            if self.diffs[0] > 0.001 or self.diffs[1] > 1.0 or self.diffs[2] > 200.:
                print(self.diffs)
                print("difference too much")
                self.not_detected_counter = self.not_detected_counter + 1
            else:
                self.detected = True

                self.not_detected_counter = 0
                self.current_fit.append(lane_fit)
                if len(self.current_fit) > 5:
                    self.current_fit = self.current_fit[len(self.current_fit)-5:]
                self.best_fit = np.average(self.current_fit, axis=0)
                self.calc_curvature()
        else:
            self.not_detected_counter = self.not_detected_counter + 1

        if self.not_detected_counter >= 2:
            self.detected = False
            self.best_fit = None
            self.diffs = np.array([0, 0, 0], dtype='float')
            self.current_fit = []
            self.not_detected_counter = 0
            print("reset")
