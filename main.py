import numpy as np
import cv2


class RubiksColor:
    def __init__(self, id, color_range, visualization_color):
        self.id = id
        self.color_range = color_range
        self.visualization_color = visualization_color

    id: str
    color_range: tuple
    visualization_color: tuple


class FoundSquare:
    def __init__(self, contour, color):
        self.contour = contour
        self.color = color
        self.number = 0

    contour: np.ndarray
    color: RubiksColor
    number: int


class CubeOrientation:
    def __init__(self, origin, cw_next, cw_prev):
        self.origin = origin
        self.cw_next = cw_next
        self.cw_prev = cw_prev

    origin: cv2.Mat
    cw_next: cv2.Mat
    cw_prev: cv2.Mat


rubiks_colors = [
    RubiksColor(
        id="red",
        color_range=[(0, 100, 70), (10, 255, 255)],
        visualization_color=(0, 0, 255)
    ),
    RubiksColor(
        id="green",
        color_range=[(50, 100, 70), (100, 255, 255)],
        visualization_color=(0, 255, 0)
    ),
    RubiksColor(
        id="blue",
        color_range=[(100, 100, 70), (130, 255, 255)],
        visualization_color=(255, 0, 0)
    ),
    RubiksColor(
        id="yellow",
        color_range=[(20, 100, 70), (40, 255, 255)],
        visualization_color=(0, 255, 255)
    ),
    RubiksColor(
        id="orange",
        color_range=[(10, 100, 70), (20, 255, 255)],
        visualization_color=(0, 165, 255)
    ),
    RubiksColor(
        id="white",
        color_range=[(0, 0, 100), (180, 110, 255)],
        visualization_color=(255, 255, 255)
    ),
]


def is_square(aspect_ratio):
    """
    Check if a rectangle is a square
    :param aspect_ratio: The aspect ratio of the rectangle
    :return: True if the rectangle is a square, False otherwise
    """
    return 0.8 < aspect_ratio < 1.2


def rectify_contour(contour):
    """
    Rectify a contour by finding the smallest rectangle that contains the contour
    :param contour: Contour to rectify
    :return: The rectified contour and its aspect ratio
    """
    rotated_rect = cv2.minAreaRect(contour)
    aspect_ratio = rotated_rect[1][0] / rotated_rect[1][1]

    box = cv2.boxPoints(rotated_rect)
    box = np.intp(box)

    return box, aspect_ratio


def find_contours_with_color_range(image, color_range):
    """
    Find all contours in an image that are within a certain color range
    :param image: Image to find contours in
    :param color_range: Color range to find contours in
    :return: A list of found contours
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_color, upper_color = color_range

    color_mask = cv2.inRange(hsv_image, lower_color, upper_color)
    contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def find_square_contours(contours):
    """
    Find all square contours in a list of contours
    :param contours: A list of contours
    :return: A list of found square contours
    """
    square_contours = []

    for contour in contours:
        # Ignore very small contours
        contour_area = cv2.contourArea(contour)

        if contour_area < 100:
            continue

        # Create a bounding box around the contour
        # If the area ratio between the bounding box and the found contour is more than 200%, we know
        # that the contour is definitely not a square
        center, size, _ = cv2.minAreaRect(contour)
        bounding_box_area = size[0] * size[1]

        if bounding_box_area / contour_area > 1.2:
            continue

        # Create a more simple contour py reducing the number of points
        max_distance_from_original_curve_px = 0.03 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, max_distance_from_original_curve_px, True)

        # Count the number of edges in the simplified contour (filter out small edges)
        num_edges = 0

        for i in range(len(contour)):
            point = contour[i]
            next_point = contour[(i + 1) % len(contour)]

            distance = np.linalg.norm(point - next_point)

            if distance > 10:
                num_edges += 1

        # If the contour has less than 4 edges, it is not a square
        # If the contour has much more than 4 edges, it is not a square
        if num_edges < 4 or num_edges > 6:
            continue

        # Rectify the contour
        # If final aspect ratio is not close to 1, it is not a square
        contour, aspect_ratio = rectify_contour(contour)
        if aspect_ratio < 0.8 or aspect_ratio > 1.2:
            continue

        # We found a square!!
        square_contours.append(contour)

    return square_contours


def find_cube_contour(found_squares):
    """
    Based on the found squares, find the contour of the entire Rubik's Cube
    :param found_squares: The found squares
    :return: Contour of cube
    """
    # If we do not have 9 squares, we cannot have a cube
    if len(found_squares) != 9:
        return None

    # Get the total contour of all squares together with
    # the help of the convex hull algorithm
    all_contours = []
    for square in found_squares:
        all_contours.append(square.contour)

    cube_contour = cv2.convexHull(
        np.concatenate(all_contours)
    )

    # Rectify the cube contour and check if it is a square
    cube_contour, aspect_ratio = rectify_contour(cube_contour)

    if not is_square(aspect_ratio):
        return None

    return cube_contour


def find_cube_orientation(cube_contour):
    """
    Decide the orientation of the cube depending on which side is pointing upwards the most
    Three corners of the cube will be labeled as origin, clockwise next and clockwise previous:
    o=origin
    cn=clockwise next
    cp=clockwise previous
    o---cn
    |   |
    cp--.
    :param cube_contour: The contour of the cube
    :return: The orientation of the cube (as a CubeOrientation object)
    """
    y_sorted_contour = cube_contour[cube_contour[:, 1].argsort()]
    x_sorted_contour = cube_contour[cube_contour[:, 0].argsort()]

    top_points = y_sorted_contour[:2]
    leftmost_points = x_sorted_contour[:2]

    origin_point, clockwise_next_point = top_points[top_points[:, 0].argsort()]
    clockwise_previous_point = next(p for p in leftmost_points if not np.array_equal(p, origin_point))

    orientation = CubeOrientation(
        origin=origin_point,
        cw_next=clockwise_next_point,
        cw_prev=clockwise_previous_point
    )

    return orientation


def number_found_squares(frame, orientation, found_squares):
    # Calculate the side length and tilt of the cube
    origin_point = orientation.origin
    cw_next_point = orientation.cw_next
    cw_prev_point = orientation.cw_prev

    cube_tilt_k_value = (cw_next_point[1] - origin_point[1]) / (cw_next_point[0] - origin_point[0])
    side_length = np.linalg.norm(cw_next_point - origin_point)

    for i in range(3):
        # Calculate column position x-wise relative to Cube's coordinate system
        x = i + 1
        cube_x = x * side_length / 3 - side_length / 6

        # Calculate row start and stop position relative to image's coordinate system
        theta = np.arctan2(cube_tilt_k_value, 1)

        x_diff = cube_x * np.cos(theta)
        y_diff = cube_x * np.sin(theta)

        x1 = int(origin_point[0] + x_diff)
        y1 = int(origin_point[1] + y_diff)

        x2 = int(cw_prev_point[0] + x_diff)
        y2 = int(cw_prev_point[1] + y_diff)

        cv2.line(frame, (x1, y1), (x2, y2), (121, 131, 248), 2)

        # Go down the column and find the squares
        searchable_squares = found_squares.copy()
        search_step_size = int(side_length / 6)

        num_found_squares = 0

        for search_y in range(y1, y2, search_step_size):
            search_x = np.interp(search_y, [y1, y2], [x1, x2])
            search_x = int(search_x)

            cv2.circle(frame, (search_x, search_y), 1, (0, 0, 0), 2)

            for j, square in enumerate(searchable_squares):
                intersects_square = cv2.pointPolygonTest(square.contour, (search_x, search_y), False) >= 0

                if intersects_square:
                    square_number = i + 1 + 3 * num_found_squares
                    found_squares[j].number = square_number

                    num_found_squares += 1
                    del searchable_squares[j]

                    cv2.putText(
                        frame, str(square_number), (search_x, search_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )
                    break


def find_rubiks_side(frame):
    """
    Try to find the side of a Rubik's Cube in a frame
    Including all 9 squares and their colors
    :param frame: Image frame
    :return:
    """
    found_squares = []

    # Search for squares in the image for each color
    for rubiks_color in rubiks_colors:
        color_range = rubiks_color.color_range
        visualization_color = rubiks_color.visualization_color

        found_contours = find_contours_with_color_range(frame, color_range)
        square_contours = find_square_contours(found_contours)

        cv2.drawContours(frame, square_contours, -1, visualization_color, 3)

        for square_contour in square_contours:
            found_squares.append(FoundSquare(square_contour, rubiks_color))

    # Find the contour of the entire Rubik's Cube
    cube_contour = find_cube_contour(found_squares)

    if cube_contour is None:
        return

    cv2.drawContours(frame, [cube_contour], 0, (99, 49, 222), 2)

    # Find cube orientation
    cube_orientation = find_cube_orientation(cube_contour)

    cv2.circle(frame, tuple(cube_orientation.origin), 5, (255, 0, 0), 2)
    cv2.circle(frame, tuple(cube_orientation.cw_next), 5, (0, 255, 0), 2)
    cv2.circle(frame, tuple(cube_orientation.cw_prev), 5, (0, 0, 255), 2)

    # Number the squares
    number_found_squares(frame, cube_orientation, found_squares)

    return True


def process_frame(frame):
    found_squares = find_rubiks_side(frame)

    if found_squares is not None:
        cv2.putText(frame, "Found Rubik's Cube", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0))
    else:
        cv2.putText(frame, "No Rubik's Cube found", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))

    return frame


def main():
    # Capture video
    cap = cv2.VideoCapture(0)

    # Show video in a window
    while True:
        ret, frame = cap.read()

        processed = process_frame(frame)
        cv2.imshow('frame', processed)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
