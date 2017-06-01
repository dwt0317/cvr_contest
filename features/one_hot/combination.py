# -*- coding:utf-8 -*-


# 生成统计类特征
def build_high_combination(connectionType,  appPlatform, appCategory, sitesetID,
                      positionType, gender, education,
                      marriageStatus, haveBaby, age, hometown, residence):
    idx = 0
    features = [0] * 11
    if connectionType == 1:
        if positionType == 5:
            features[idx] = 1
        idx += 1

    if sitesetID == 1:
        if positionType == 5:
            features[idx] = 1
        idx += 1
        if education == 2:
            features[idx] = 1
        idx += 1
        if marriageStatus == 3:
            features[idx] = 1
        idx += 1

    if positionType == 0:
        if education == 2:
            features[idx] = 1
        idx += 1
        if residence in [24, 15]:
            features[idx] = 1
        idx += 1
    if appCategory == 4:
        if sitesetID == 1:
            features[idx] = 1
        idx += 1
        if education in [0, 2, 3, 4]:
            features[idx] = 1
        idx += 1
        if age in [0, 9]:
            features[idx] = 1
        idx += 1
    if appCategory == 5:
        if education == 1:
            features[idx] = 1
        idx += 1
    if appCategory == 1:
        if haveBaby == 4:
            features[idx] = 1
        idx += 1
    return features

def build_low_combination(connectionType, appPlatform, appCategory, sitesetID,
                      positionType, gender, education,
                      marriageStatus, haveBaby, age, hometown, residence):
    idx = 0
    features = [0] * 13
    if connectionType != 1:
        if sitesetID == 0:
            features[idx] = 1
        idx += 1
        if positionType in [1, 2, 3]:
            features[idx] = 1
        idx += 1
        if haveBaby in [0, 1]:
            features[idx] = 1
        idx += 1
        if age in [0, 9]:
            features[idx] = 1
        idx += 1
    if appPlatform == 2:
        if sitesetID == 0:
            features[idx] = 1
        idx += 1
    if sitesetID == 0:
        if positionType in [0, 2, 4]:
            features[idx] = 1
        idx += 1
    if positionType == 2:
        if gender == 2:
            features[idx] = 1
        idx += 1
        if education in [1, 3, 4, 5]:
            features[idx] = 1
        idx += 1
        if haveBaby in [0, 1]:
            features[idx] = 1
        idx += 1
    if gender == 0:
        if residence == 13:
            features[idx] = 1
        idx += 1
    if appCategory == 4:
        if sitesetID == 2:
            features[idx] = 1
        idx += 1
    return features