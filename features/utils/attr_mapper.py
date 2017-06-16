# -*- coding:utf-8 -*-


# 属性映射dict
def build_attr_map(ad_map, user_map, userID, creativeID, positionID, connectionType):
    age, gender, education, marriageStatus, haveBaby, hometown, residence = user_map
    a_n = len(ad_map[creativeID])
    appPlatform, appCategory, appID = ad_map[creativeID][a_n - 2], ad_map[creativeID][a_n - 1], ad_map[creativeID][
        a_n - 3]
    campaignID, adID = ad_map[creativeID][a_n - 5], ad_map[creativeID][a_n - 6]
    advertiserID = ad_map[creativeID][a_n - 4]
    attr_map = {}
    attr_map['userID'] = userID
    attr_map['age'] = age
    attr_map['gender'] = gender
    attr_map['education'] = education
    attr_map['marriageStatus'] = marriageStatus
    attr_map['haveBaby'] = haveBaby
    attr_map['hometown'] = hometown
    attr_map['residence'] = residence
    attr_map['appPlatform'] = appPlatform
    attr_map['appCategory'] = appCategory
    attr_map['appID'] = appID
    attr_map['campaignID'] = campaignID
    attr_map['adID'] = adID
    attr_map['advertiserID'] = advertiserID
    attr_map['creativeID'] = creativeID
    attr_map['positionID'] = positionID
    attr_map['connectionType'] = connectionType
    return attr_map

