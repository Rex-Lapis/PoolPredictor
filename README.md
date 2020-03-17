# PoolPredictor
Looks at overhead video of billiards table, detects boundaries and balls, and predicts the angles and motion of balls. 

## Not finished and under construction!
### Currently, it will play the video, circling each ball with it's color, and draw trajectories when the ball starts moving. It currently does a decent job with single bounce reflections, but the bounce algorithm still needs to be made recursive. OpenGL is required to acheive necessary speed, and so it may not run out of the box.


![Demo Gif](/doc_resources/PoolGif1.gif)


# The Algorithm:
## 1. Take a few frames, and do analysis for straight lines with the correct distance ratios and relative angles as a pool table would.
## 2. Separate found lines into bumper, pocket, and table edge lines.
## 3. Identify pocket areas based on line intersections and lengths between intersections.
## 4. Start checking frames live, looking only within the table boundaries for circles.
## 5. Keep only circles with radii within plausable range for a billiard ball
## 6. Check the average color within each circle against that of the table cloth. if it's within a certain threshold of similarity, throw it out.
## 7. For the first few frames, we append all circles found to a frame-buffer of balls
## Not done writing this out...
