import numpy as np
import cv2
import math

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
W = 7
H = 5
square_pixel_size = 0

def get_edges(img):
    kernel = np.ones((3,3),np.uint8)
    edges = cv2.Canny(img, 0, 255)
    edges = cv2.dilate(edges,kernel,iterations = 1)
    return edges

def get_circlies(img):
    ball_pixel_size = 68 / 2 * square_pixel_size / 30
    circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        1,
        50,
        param1=50,
        param2=30,
        minRadius=int(ball_pixel_size) - 10,
        maxRadius=int(ball_pixel_size) + 10
    )
    res = np.squeeze(circles)
    for c in res:
        c[2] = ball_pixel_size
    return res

def fix_perspective(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (W,H),None)

    cv2.cornerSubPix(gray,corners,(7,7),(-1,-1),criteria)
    img = cv2.drawChessboardCorners(img, (W,H), corners,ret)

    corners = np.float32([[x[0,0], x[0, 1]] for x in corners])

    wavg = 0
    for i in range(0, W):
        wavg += np.linalg.norm(corners[i] - corners[28 + i])
    wavg /= W * (H - 1)

    havg = 0
    for i in range(0, H):
        havg += np.linalg.norm(corners[i * 7] - corners[i * 7 + 6])
    havg /= H * (W - 1)

    step = int(math.ceil((wavg + havg) / 2))
    global square_pixel_size
    square_pixel_size = step
    corners = np.float32([corners[0], corners[28], corners[34], corners[6]])

    h, w, _ = img.shape
    new_corners = np.float32([
        [w - 8 * step, h - 6 * step],
        [w - 8 * step, h - 2 * step],
        [w - 2 * step, h - 2 * step],
        [w - 2 * step, h - 6 * step],
    ])

    M = cv2.getPerspectiveTransform(corners, new_corners)

    for i in corners:
        cv2.circle(img,(int(i[0]),int(i[1])), 4,(255,0,255),6)

    dst = cv2.warpPerspective(img,M,(w,h))

    return dst

def clean_up_image(dst):
    Z = np.float32(dst.reshape((-1,3)))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    _,labels,centers = cv2.kmeans(Z, K, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)

    white_label = np.argmax(np.all(centers > 200, axis=1))
    labels = labels.reshape((img.shape[:-1]))
    cleared = ((labels == white_label) * 255).astype(np.uint8)

    h, w, _ = dst.shape
    cv2.rectangle(cleared,
        (w - (W + 3) * square_pixel_size, h - (H + 3) * square_pixel_size),
        (w - 1, h - 1),
        0, -1
    )

    return cleared

def draw_circles(dst, circles):
    for i in np.uint16(np.around(circles)):
        cv2.circle(dst, (i[0], i[1]), i[2], (0,0,255), 7)

def get_cue_direction(edges):
    lines = cv2.HoughLines(edges,1,np.pi/180, 200) 

    lines = np.squeeze(lines)

    avg_line = lines.mean(axis=0)

    def get_line_coeffs(r, thetta):
        c = np.cos(thetta)
        s = np.sin(thetta)
        return -c/s, r/s

    return get_line_coeffs(avg_line[0], avg_line[1])

def cofs_to_image_points(cue_a, cue_b, dst):
    def solve4y(a, b, y):
        return (y - b) / a, y

    def solve4x(a, b, x):
        return x, a * x + b

    h, w, _ = dst.shape
    def coefs_to_line(a, b):
        points = []
        points.append(solve4x(a, b, 0))
        points.append(solve4x(a, b, w - 1))
        points.append(solve4y(a, b, 0))
        points.append(solve4y(a, b, h - 1))
        def in_image(p):
            return 0 <= p[0] < w and 0 <= p[1] < h
        points = list(filter(in_image, points))
        return points

    return coefs_to_line(cue_a, cue_b)

def lin_equ(l1, l2):
    m = ((l2[1] - l1[1])) / (l2[0] - l1[0])
    c = (l2[1] - (m * l2[0]))
    return m, c

def draw_direction(start_p, end_p, dst, color):
    cue_a, cue_b = lin_equ(start_p, end_p)
    image_points = cofs_to_image_points(cue_a, cue_b, dst)
    x1, y1 = start_p
    x2, y2 = min(image_points, key = lambda c: np.linalg.norm(np.array([c[0], c[1]]) - [end_p[0], end_p[1]]))
    cv2.line(dst,(int(x1),int(y1)),(int(x2),int(y2)),color,5)

def get_orth_coefs(x, y, a):
    return -1 / a, x/a + y

def find_cross_point(a1, b1, a2, b2):
    A = np.array([
        [a1, -1],
        [a2, -1],
    ])
    B = np.array([-b1, -b2])
    X = np.linalg.inv(A).dot(B)
    return X[0], X[1]

def project_to_line(a, b, x, y):
    a1, b1 = get_orth_coefs(x, y, a)
    return find_cross_point(a, b, a1, b1)

def find_circles_projections(cue_a, cue_b, circles, dst):
    res = []
    for i in circles:
        x1, y1 = project_to_line(cue_a, cue_b, i[0], i[1])
        res.append([x1, y1])
        cv2.circle(dst, (int(x1),int(y1)), 5, (0,0,255), 7)

    return res

def find_cue_end_point(edges, cue_a, cue_b, circles, dst):
    ttt = cv2.cornerHarris(edges,10,5,0.04)

    #result is dilated for marking the corners, not important
    ttt = cv2.dilate(ttt,None)
    ret, ttt = cv2.threshold(ttt,0.01*dst.max(),255,0)
    ttt = np.uint8(ttt)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(ttt)

    #define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(edges,np.float32(centroids),(5,5),(-1,-1),criteria)

    def in_circle(c, r, p):
        x = p[0] - c[0]
        y = p[1] - c[1]
        return x * x + y * y < r * r

    def not_remove_point(p):
        for c in circles:
            if in_circle(c, c[2], p):
                return False
        return True

    corners = list(filter(not_remove_point, corners.tolist()))

    def dist_to_circle(p):
        dists = []
        for c in circles:
            dists.append(np.linalg.norm(np.array([c[0], c[1]]) - p))
        return min(dists)

    closest_point  = min(corners, key=dist_to_circle)
    cue_end_point = project_to_line(cue_a, cue_b, closest_point[0], closest_point[1])

    cv2.circle(dst, (int(cue_end_point[0]), int(cue_end_point[1])), 5, (255,0,255), 7)

    return cue_end_point

def move_between_points(delta, p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist = np.sqrt(dx*dx + dy*dy)

    alfa = delta/ dist
    return [
        p1[0] + alfa * dx,
        p1[1] + alfa * dy
    ]

img     = cv2.imread('./in/src_balls.jpg')
dst     = fix_perspective(img)
cleared = clean_up_image(dst)
edges   = get_edges(cleared)

cv2.imwrite('./out/step.jpg',  img)
cv2.imwrite('./out/step1.jpg', dst)
cv2.imwrite('./out/step2.jpg', cleared)
cv2.imwrite('./out/step3.jpg', edges)

circles = get_circlies(edges)
draw_circles(dst, circles)

cue_a, cue_b = get_cue_direction(edges)

cue_end_point = find_cue_end_point(edges, cue_a, cue_b, circles, dst)

circles = sorted(circles, key = lambda c: np.linalg.norm(np.array([c[0], c[1]]) - cue_end_point))
circles_projections = find_circles_projections(cue_a, cue_b, circles, dst)

draw_direction(cue_end_point, circles_projections[1], dst, [255, 0, 255])

cue_ball      = circles[0]
cue_ball_proj = circles_projections[0]
target_ball   = circles[1]
cv2.circle(dst, (int(cue_ball[0]), int(cue_ball[1])), 5, (0,255,0), 7)
cv2.circle(dst, (int(target_ball[0]), int(target_ball[1])), 5, (255,0,0), 7)

cue_ball_mid_proj_dist = np.linalg.norm(np.array([cue_ball[0], cue_ball[1]]) - cue_ball_proj)
delta = np.sqrt(cue_ball[2]**2 - cue_ball_mid_proj_dist**2)

cue_ball_new_pos  = move_between_points(target_ball[2] + cue_ball[2], target_ball, cue_ball)
cv2.circle(dst, (int(cue_ball_new_pos[0]), int(cue_ball_new_pos[1])), int(cue_ball[2]), (0,255,255), 7)

cue_and_ball_touch_point = move_between_points(delta, cue_ball_proj, cue_end_point)
cv2.circle(dst, (int(cue_and_ball_touch_point[0]), int(cue_and_ball_touch_point[1])), 4, (255,255,0), 5)

balls_touch_point = move_between_points(target_ball[2], target_ball, cue_ball)
cv2.circle(dst, (int(balls_touch_point[0]), int(balls_touch_point[1])), 4, (0,255,0), 5)

draw_direction(cue_and_ball_touch_point, np.array([cue_ball[0], cue_ball[1]]), dst, [255, 255, 0])

draw_direction(balls_touch_point, np.array([target_ball[0], target_ball[1]]), dst, [0, 255, 0])

cv2.imwrite('./out/result.jpg', dst)
