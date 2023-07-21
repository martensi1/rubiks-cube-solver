import numpy as np
import cv2


class RubiksColor:
    def __init__(self, name, color_range, visualization_color):
        self.name = name
        self.color_range = color_range
        self.visualization_color = visualization_color

    name: str
    color_range: tuple
    visualization_color: tuple


rubiks_colors = [
    RubiksColor(
        name="red",
        color_range=[(0, 100, 70), (10, 255, 255)],
        visualization_color=(0, 0, 255)
    ),
    RubiksColor(
        name="green",
        color_range=[(50, 100, 70), (100, 255, 255)],
        visualization_color=(0, 255, 0)
    ),
    RubiksColor(
        name="blue",
        color_range=[(100, 100, 70), (130, 255, 255)],
        visualization_color=(255, 0, 0)
    ),
    RubiksColor(
        name="yellow",
        color_range=[(20, 100, 70), (40, 255, 255)],
        visualization_color=(0, 255, 255)
    ),
    RubiksColor(
        name="orange",
        color_range=[(10, 100, 70), (20, 255, 255)],
        visualization_color=(0, 165, 255)
    ),
    RubiksColor(
        name="white",
        color_range=[(0, 0, 100), (180, 110, 255)],
        visualization_color=(255, 255, 255)
    ),
]


def rectify_contour(contour):
    """
    Rectify a contour by finding the smallest rectangle that contains the contour
    :param contour: Contour to rectify
    :return: The rectified contour and its aspect ratio
    """
    rotated_rect = cv2.minAreaRect(contour)
    aspect_ratio = rotated_rect[1][0] / rotated_rect[1][1]

    box = cv2.boxPoints(rotated_rect)
    box = np.int0(box)

    return box, aspect_ratio


def find_contours_with_color_range(image, min_color, max_color):
    """
    Find all contours in an image that are within a certain color range
    :param image: Image to find contours in
    :param min_color: Lower bound of the color range
    :param max_color: Upper bound of the color range
    :return: A list of found contours
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    color_mask = cv2.inRange(hsv_image, min_color, max_color)
    contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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

        # x1, y1, w, h = cv2.boundingRect(contour)
        # cv2.putText(frame, str(len(contour)) + "/" + str(len(side_lengths)), (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

    return square_contours


def number_found_squares(frame, found_squares):
    # Get the total contour of all squares together
    side_contour = cv2.convexHull(
        np.concatenate(found_squares)
    )

    # Convert found contour to a rectangle
    side_rect_contour, _ = rectify_contour(side_contour)
    cv2.drawContours(frame, [side_rect_contour], 0, (0, 0, 255), 2)

    # Find the orientation (origin, clockwise next point, clockwise previous point) of the rectangle
    y_sorted_contour = side_rect_contour[side_rect_contour[:, 1].argsort()]
    x_sorted_contour = side_rect_contour[side_rect_contour[:, 0].argsort()]

    top_points = y_sorted_contour[:2]
    leftmost_points = x_sorted_contour[:2]

    origin_point, clockwise_next_point = top_points[top_points[:, 0].argsort()]
    clockwise_previous_point = next(p for p in leftmost_points if not np.array_equal(p, origin_point))

    cv2.circle(frame, tuple(origin_point), 5, (255, 0, 0), 2)
    cv2.circle(frame, tuple(clockwise_next_point), 5, (0, 255, 0), 2)
    cv2.circle(frame, tuple(clockwise_previous_point), 5, (0, 0, 255), 2)

    # Estimate square positions based on the orientation of the rectangle
    row_k_value = (clockwise_next_point[1] - origin_point[1]) / (clockwise_next_point[0] - origin_point[0])
    column_k_value = (clockwise_previous_point[1] - origin_point[1]) / (clockwise_previous_point[0] - origin_point[0])

    row_length = np.linalg.norm(clockwise_next_point - origin_point)
    column_length = np.linalg.norm(clockwise_previous_point - origin_point)

    for i in range(9):
        x = i % 3 + 1
        y = i // 3 + 1

        # Calculate square position relative to the Cube's coordinate system
        cube_x = x * row_length / 3 - row_length / 6
        cube_y = y * column_length / 3 - column_length / 6

        # Calculate square position relative to the image's coordinate system
        theta = np.arctan2(row_k_value, 1)

        x = cube_x * np.cos(theta)
        y = cube_x * np.sin(theta)

        # theta = np.arctan2(column_k_value, 1)

        # x += cube_y * np.cos(theta)
        # y += cube_y * np.sin(theta)

        actual_x = origin_point[0] + x
        actual_y = origin_point[1] + y

        actual_x2 = clockwise_previous_point[0] + x
        actual_y2 = clockwise_previous_point[1] + y

        cv2.line(frame, (int(actual_x), int(actual_y)), (int(actual_x2), int(actual_y2)), (0, 0, 0), 2)

        # cv2.circle(frame, (int(actual_x), int(actual_y)), 5, (0, 0, 0), 2)


def find_rubiks_side(frame):
    """
    Try to find the side of a Rubik's Cube in a frame
    Including all 9 squares and their colors
    :param frame: Image frame
    :return:
    """
    found_squares = []

    # Denoise image
    de_noised_image = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

    # Search for squares in the image for each color
    for rubiks_color in rubiks_colors:
        min_color, max_color = rubiks_color.color_range
        found_contours = find_contours_with_color_range(de_noised_image, min_color, max_color)

        square_contours = find_square_contours(found_contours)
        cv2.drawContours(frame, square_contours, -1, rubiks_color.visualization_color, 3)

        for i, square_contour in enumerate(square_contours):
            x, y, w, h = cv2.boundingRect(square_contour)

            inverted_color = (
                255 - rubiks_color.visualization_color[0],
                255 - rubiks_color.visualization_color[1],
                255 - rubiks_color.visualization_color[2]
            )

            cv2.putText(frame, str(i), (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1.0, inverted_color)

        found_squares += square_contours

    # If we did not find all 9 squares, return None
    if len(found_squares) != 9:
        return None

    # Number the squares
    number_found_squares(frame, found_squares)

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

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
