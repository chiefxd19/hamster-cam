
import cv2
import mediapipe as mp
import numpy as np
import os

mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

SPRITE_DIR = os.path.join(os.path.dirname(__file__), "sprites")

def load_sprites():
    sprites = {}
    for name in ["neutral","smile","laugh","thumbs_down","thumbs_up","peace","tutu","nerd","pleading"]:
        p = os.path.join(SPRITE_DIR, f"{name}.png")
        if os.path.exists(p):
            sprites[name] = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    return sprites

def overlay_transparent(bg, overlay):
    h,w = bg.shape[:2]
    oh,ow = overlay.shape[:2]
    scale = min(w/ow, h/oh) * 0.85
    if scale < 1.0 or scale > 1.0:
        overlay = cv2.resize(overlay, (int(ow*scale), int(oh*scale)), interpolation=cv2.INTER_AREA)
    oh,ow = overlay.shape[:2]
    x, y = (w-ow)//2, (h-oh)//2
    if overlay.shape[2]==4:
        alpha = overlay[:,:,3].astype(float)/255.0
        alpha = np.stack([alpha]*3, axis=-1)
        roi = bg[y:y+oh, x:x+ow].astype(float)
        ov_rgb = overlay[:,:,:3].astype(float)
        bg[y:y+oh, x:x+ow] = (roi*(1-alpha) + ov_rgb*alpha).astype(np.uint8)
    else:
        bg[y:y+oh, x:x+ow] = overlay[:,:,:3]
    return bg

def get_gesture(hand_lm):
    lm = hand_lm.landmark
    tips = [8,12,16,20]
    pips = [6,10,14,18]
    up = [1 if lm[t].y < lm[p].y -0.02 else 0 for t,p in zip(tips,pips)]
    thumb_tip, thumb_ip = lm[4], lm[3]
    thumb_down = thumb_tip.y > thumb_ip.y + 0.04 and thumb_tip.x < thumb_ip.x
    thumb_up = thumb_tip.y < thumb_ip.y -0.04

    if sum(up)==0 and thumb_down:
        return "thumbs_down", 5
    if sum(up)==0 and thumb_up:
        return "thumbs_up", 4
    if up == [0,1,0,0] or up == [0,1,1,0]:
        return "peace", 12
    if sum(up)>=3 and up[0]==1:
        return "tutu", 4
    if up == [1,0,0,1]:
        return "nerd", 14
    return "neutral", 13

def get_face(lm):
    mouth_open = abs(lm.landmark[13].y - lm.landmark[14].y)
    smile = abs(lm.landmark[61].x - lm.landmark[291].x)
    if mouth_open > 0.06:
        return "laugh", 7
    if smile > 0.09:
        return "smile", 8
    return "neutral", 15

SPRITES = load_sprites()
cap = cv2.VideoCapture(0)

with mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh,      mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6) as hands:

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h,w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_res = face_mesh.process(rgb)
        hand_res = hands.process(rgb)

        state = "neutral"
        fid = 15

        if face_res.multi_face_landmarks:
            for fl in face_res.multi_face_landmarks:
                state, fid = get_face(fl)

        if hand_res.multi_hand_landmarks:
            for hl in hand_res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                gname, gid = get_gesture(hl)
                if gname != "neutral":
                    state, fid = gname, gid

        # render hamster
        hamster_bg = np.ones((h,w,3), dtype=np.uint8)*255
        if state in SPRITES:
            hamster_bg = overlay_transparent(hamster_bg, SPRITES[state])
        else:
            hamster_bg = overlay_transparent(hamster_bg, SPRITES["neutral"])

        label = f"#{fid} ({state}) NK1@{np.random.uniform(0.1,0.9):.2f}"
        cv2.rectangle(hamster_bg, (5, h-30), (260, h-5), (0,0,0), -1)
        cv2.putText(hamster_bg, label, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        combined = np.hstack([frame, hamster_bg])
        cv2.imshow("Hamster Mirror - Traced Pack", combined)
        if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
