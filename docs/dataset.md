# Observation
- The observation contains three RGB images and three depth images. There are three cameras.
- After self.get_image(obs) it will return a (320, 160, 6) image. The first three channels are normalized RGB image. The last three channel are heightmaps.