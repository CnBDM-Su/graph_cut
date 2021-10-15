import numpy as np

def combine(face_pred_path, body_pred_path, face_quality_path, face_quality_thres, face_body_edge_path, face_body_pred_path):

    face_label = np.load(face_pred_path)
    body_label = np.load(body_pred_path)
    face_qulity = np.load(face_quality_path)

    fb_edge = np.load(face_body_edge_path)
    fb_track = {}
    for i in range(fb_edge.shape[0]):
        i = fb_edge[:,i]
        if i[1] not in fb_track:
            fb_track[i[1]] = [i[0]]
        else:
            fb_track[i[1]].append(i[0])

    face_profile = {}
    for i in range(face_label.shape[0]):
        if face_label[i] not in face_profile.keys():
            face_profile[face_label[i]] = [i]
        else:
            face_profile[face_label[i]].append(i)
    face_profile = list(face_profile.values())\

    body_profile = {}
    for i in range(body_label.shape[0]):
        if body_label[i] not in body_profile.keys():
            body_profile[body_label[i]] = [i]
        else:
            body_profile[body_label[i]].append(i)

    unique_body = np.unique(body_label)
    unique_face = np.unique(face_label)
    profile_edge = np.zeros(shape=(unique_face.shape[0], unique_body.shape[0]))
    b_lis = {}
    f_lis = {}

    for i in range(unique_body.shape[0]):
        b_lis[unique_body[i]] = i
    for i in range(unique_face.shape[0]):
        f_lis[unique_face[i]] = i

    for i in range(fb_edge.shape[1]):
        profile_edge[f_lis[int(face_label[int(fb_edge[0,i])])], b_lis[int(body_label[int(fb_edge[1,i]-face_label.shape[0])])]] += 1

    #quality filter
    ignore_face = []
    for i in face_profile.keys():
        mean_ = np.mean(face_quality[face_profile[i]])
        if mean_ < face_quality_thres:
            ignore_face.append(i)

    for i in ignore_face:
        profile_edge[int(i),:] = 0

    body_label += unique_face.shape[0]
    body_relate_max = np.max(profile_edge, axis=0)
    body_relate = np.argmax(profile_edge, axis=0)
    
    for i in range(body_relate.shape[0]):
        if (body_relate_max[i] > 0) and (unique_body[i] != -1):
            for j in body_profile[unique_body[i]]:
                body_label[j] = body_relate[i]

    pred = np.concatenate([face_label, body_label],0)
    np.save(face_body_pred_path, pred)