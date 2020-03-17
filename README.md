# PoolPredictor
Looks at overhead video of billiards table, detects boundaries and balls, and predicts the angles and motion of balls. 

## Not finished and under construction!
### Currently, it will play the video, circling each ball with it's color, and draw trajectories when the ball starts moving. It currently does a decent job with single bounce reflections, but the bounce algorithm still needs to be made recursive. The reflection algorithm also needs to be tweaked so that the reflection vector doesn't grow larger as the ball approaches the wall. Balls turn red when they are going to be hit by a ball in motion. This area still needs a little work as well. OpenGL is required to acheive necessary speed, and so it may not run out of the box.

### Example:
![Demo Gif](/doc_resources/PoolGif1.gif)


# The Algorithm:
## 1. Take a few frames, and do analysis for straight lines with the distance ratios and relative angles as a pool table would have.
## 2. Separate found lines into bumper, pocket, and table edge lines.
## 3. Identify pocket areas based on line intersections and lengths between intersections.
## 4. Start checking frames live, looking only within the table boundaries for circles.
## 5. Keep only circles with radii within plausable range for a billiard ball
## 6. Check the average color within each circle against that of the table cloth. if it's within a certain threshold of similarity, throw it out.
## 7. It keeps a fixed length FIFO buffer of frames (currently 5) where ball qualified objects are stored.
## 8. With each new frame each new circle in that frame is checked against the balls objects in the frame buffer for color-similarity, and distance. If a good match is found, the circle will be added to the frame buffer for that ball.
## 9. If circles being added to a ball start having regular distance from eachother then that ball enters a 'moving' state
## 10. A line is fit to the centers of circles in the moving balls buffer, and if the line fits nicely with little deviation, then direction of motion, as well as velocity is assessed.
## 11. A peliminary velocity vector is instatiated, and it is analyzed for bumper intersections
## 12. If an intersection is detected, a reflection vector is made.
## 13. Collision detection is done by checking all balls distance from any lines of motion. If the distance is within 2r, it flashes red.
