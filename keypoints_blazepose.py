"""Use Blazepose to detect 33 keypoints on image frames coming from real-time
video stream using opencv"""

# TODO: install package/repo??

from BlazeposeDepthaiEdge import BlazeposeDepthai
from BlazeposeRenderer import BlazeposeRenderer

# The argparse stuff has been removed to keep only the important code

tracker = BlazeposeDepthai(input_src=args.input, 
            pd_model=args.pd_m,
            lm_model=args.lm_m,
            smoothing=not args.no_smoothing,   
            xyz=args.xyz,           
            crop=args.crop,
            internal_fps=args.internal_fps,
            internal_frame_height=args.internal_frame_height,
            force_detection=args.force_detection,
            stats=args.stats,
            trace=args.trace)   

renderer = BlazeposeRenderer(
                pose, 
                show_3d=args.show_3d, 
                output=args.output)

while True:
    # Run blazepose on next frame
    frame, body = tracker.next_frame()
    if frame is None: break
    # Draw 2d skeleton
    frame = renderer.draw(frame, body)
    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord('q'):
        break
renderer.exit()
tracker.exit()