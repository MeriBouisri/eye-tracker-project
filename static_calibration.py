import tkinter as tk
import time
import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Create a full-screen window
root = tk.Tk()
root.attributes('-fullscreen', True)

# Create a canvas to display points
canvas = tk.Canvas(root, bg='black', highlightthickness=0)
canvas.pack(fill=tk.BOTH, expand=True)

# Calculate point size based on screen size
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
point_size = min(screen_width, screen_height) // 30
point_size2 = point_size // 2

points = []
points2 = []
point_index = 0 
all_points = []


pointscoords = [
    {'rx': [], 'ry': [], 'lx': [], 'ly': [], 'px': [], 'py': []},  # point1
    {'rx': [], 'ry': [], 'lx': [], 'ly': [], 'px': [], 'py': []},  # point2
    {'rx': [], 'ry': [], 'lx': [], 'ly': [], 'px': [], 'py': []},  # point3
    {'rx': [], 'ry': [], 'lx': [], 'ly': [], 'px': [], 'py': []},  # point4
    {'rx': [], 'ry': [], 'lx': [], 'ly': [], 'px': [], 'py': []},  # point5
    {'rx': [], 'ry': [], 'lx': [], 'ly': [], 'px': [], 'py': []},  # point6
    {'rx': [], 'ry': [], 'lx': [], 'ly': [], 'px': [], 'py': []},  # point7
    {'rx': [], 'ry': [], 'lx': [], 'ly': [], 'px': [], 'py': []},  # point8
    {'rx': [], 'ry': [], 'lx': [], 'ly': [], 'px': [], 'py': []},  # point9
    {'rx': [], 'ry': [], 'lx': [], 'ly': [], 'px': [], 'py': []},  # point10
    {'rx': [], 'ry': [], 'lx': [], 'ly': [], 'px': [], 'py': []},  # point11
    {'rx': [], 'ry': [], 'lx': [], 'ly': [], 'px': [], 'py': []},  # point12
    {'rx': [], 'ry': [], 'lx': [], 'ly': [], 'px': [], 'py': []}   # point13
]

def get_iris_coords(point_index):
    global all_points
    cam = cv2.VideoCapture(0)
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    screen_w, screen_h = pyautogui.size()

    for x in range(100):
        time.sleep(0.01)
        _, frame = cam.read()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks
        frame_h, frame_w, _ = frame.shape
        
        if landmark_points:
            landmarks = landmark_points[0].landmark
            center_right_x = 0
            center_right_y = 0
            center_left_x = 0
            center_left_y = 0

            # Right iris landmarks
            right_iris_indices = [474, 475, 476, 477]

            for id, landmark in enumerate(landmarks[474:478]):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)

                # Accumulate coordinates for center calculation
                center_right_x += x
                center_right_y += y

                if id == 1:
                    screen_right_x = screen_w * landmark.x
                    screen_right_y = screen_h * landmark.y

            # Calculate the average of the coordinates for the right eye
            center_right_x = int(center_right_x / 4)
            center_right_y = int(center_right_y / 4)
            cv2.circle(frame, (center_right_x, center_right_y), 2, (0, 0, 255), -1)

            # Left iris landmarks
            left_iris_indices = [469, 470, 471, 472]

            for id, landmark in enumerate(landmarks[469:473]):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)

                # Accumulate coordinates for center calculation
                center_left_x += x
                center_left_y += y

                if id == 1:
                    screen_left_x = screen_w * landmark.x
                    screen_left_y = screen_h * landmark.y

            # Calculate the average of the coordinates for the left eye
            center_left_x = int(center_left_x / 4)
            center_left_y = int(center_left_y / 4)
            cv2.circle(frame, (center_left_x, center_left_y), 2, (0, 0, 255), -1)

            righteye_coords = (center_right_x, center_right_y)
            lefteye_coords = (center_left_x, center_left_y)

            current_point = pointscoords[point_index]
            current_point['rx'].append(righteye_coords[0])
            current_point['ry'].append(righteye_coords[1])
            current_point['lx'].append(lefteye_coords[0])
            current_point['ly'].append(lefteye_coords[1])

            point_x, point_y = all_points[point_index]
            current_point['px'].append(point_x)
            current_point['py'].append(point_y)

    cam.release()
    cv2.destroyAllWindows()

def calculate_average():
    for i, point_data in enumerate(pointscoords):
        rx_mean = np.mean(point_data['rx'])
        ry_mean = np.mean(point_data['ry'])
        lx_mean = np.mean(point_data['lx'])
        ly_mean = np.mean(point_data['ly'])
        px = np.mean(point_data['px'])
        py = np.mean(point_data['py'])
        print(f'Point {i+1} - rx_mean: {rx_mean}, ry_mean: {ry_mean}, lx_mean: {lx_mean}, ly_mean: {ly_mean}', end=' ')
        print(f'// Coordonnées (x, y) du point {i+1} : ({px}, {py})')
    root.config(cursor='arrow')
    return all_points,pointscoords



def showpoint():
    root.config(cursor='none')

    # Create calibration points
    points = []
    points2 = []
    point_index = 0

    # Add points in each corner
    Top_left_corner = (point_size, point_size)
    Top_right_corner = (screen_width - point_size, point_size)
    Bottom_right_corner = (screen_width - point_size, screen_height - point_size)
    Bottom_left_corner = (point_size, screen_height - point_size)

    # Add center point
    center = (screen_width // 2, screen_height // 2)

    # Add points in the center of each side
    center_top = (screen_width // 2, point_size)
    center_bottom = (screen_width // 2, screen_height - point_size)
    center_left = (point_size, screen_height // 2)
    center_right = (screen_width - point_size, screen_height // 2)

    # Add four points in the center of the four corners
    center_points = [
        (screen_width // 4, screen_height // 4),  # Top-left
        (3 * screen_width // 4, screen_height // 4),  # Top-right
        (screen_width // 4, 3 * screen_height // 4),  # Bottom-left
        (3 * screen_width // 4, 3 * screen_height // 4),  # Bottom-right
    ]

    # Combine all points
    all_points = (
        [Top_left_corner]
        + [center_top]
        + [Top_right_corner]
        + [center_points[0]]
        + [center_points[1]]
        + [center_left]
        + [center]
        + [center_right]
        + [center_points[2]]
        + [center_points[3]]
        + [Bottom_left_corner]
        + [center_bottom]
        + [Bottom_right_corner]
    )

    for point in all_points:
        x, y = point

        # Create a white point
        point = canvas.create_oval(
            x - point_size, y - point_size,
            x + point_size, y + point_size,
            fill='white'
        )

        point2 = canvas.create_oval(
            x - point_size2, y - point_size2,
            x + point_size2, y + point_size2,
            fill='white'
        )
        points.append(point)
        points2.append(point2)

    return points, point_index,all_points, points2

def update_point_color():
    global points
    global points2
    global point_index
    # Check if all points are calibrated
    if point_index >= len(points):
        root.after(2000, root.destroy)  # Close the window after 2 seconds
        all_points,pointscoords = calculate_average()  # Calculate the averages

    if point_index > 0:
        canvas.itemconfig(points[point_index - 1], fill='white')
        canvas.itemconfig(points2[point_index - 1], fill='white')
    if point_index < 13:
        # Set current point color to green
        canvas.itemconfig(points[point_index], fill='green')
        canvas.itemconfig(points2[point_index], fill='red')

        get_iris_coords(point_index)

        point_index += 1

    # Schedule the next point color update after 2 seconds (adjust as needed)
    root.after(1000, update_point_color)

# Function to remove the text
def setuptext():
    global points
    global points2
    global point_index
    global all_points
    canvas.delete(text)
    points, point_index,all_points, points2 = showpoint()
    all_points = all_points
    root.after(2000, start_calibration)
    

#start updating point color
def start_calibration():
    global points
    global points2
    global point_index
    update_point_color()


# Add instructions in the center
text = canvas.create_text(
    screen_width // 2, screen_height // 2,
    text='La calibration va commencer\n        Fixez le point rouge !', fill='white', font=('Arial', 24)
)

# After 5 seconds, remove the text and start updating point color
root.after(3000, setuptext)

# Start the Tkinter event loop
root.mainloop()
