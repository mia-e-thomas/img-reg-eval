import cv2

from multipoint.utils.matching import NNMatcher

class Matcher(): 

    def __init__(self, config_matcher):

        # Save configuration
        self.config_matcher = config_matcher

        # Handle case w/ no key word arguments
        try:
            if self.config_matcher['kwargs'] is None: self.config_matcher['kwargs'] = {}
        except: 
            self.config_matcher['kwargs'] = {}

        # Choose feature constructor based on parameters
        if config_matcher['method'] == 'bf': 
            self.method = cv2.BFMatcher(**config_matcher['kwargs'])

        elif config_matcher['method'] == 'flann': 
            self.method = cv2.FlannBasedMatcher(**config_matcher['kwargs'])

        else: 
            raise ValueError('Unknown matching method: ' + config_matcher['method'])

    def match(self,des0,des1):

        # Check if ratio test requested
        if self.config_matcher['ratio_test']: 
            matches = self.method.knnMatch(des0,des1,k=2)

            # Apply ratio test
            good = []
            for m,n in matches:
                if m.distance < self.config_matcher['ratio']*n.distance:
                    good.append([m])

            return good

        # Matching w/o ratio test
        else: 
            return self.method.match(des0,des1)

