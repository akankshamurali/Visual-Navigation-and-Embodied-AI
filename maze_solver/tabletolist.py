import numpy as np
import pickle
import pandas
# Create a 2D array filled with (0,0) tuples

location_coord = np.empty((7451,2))
location_coord[930:969] =np.array([-5,0])
location_coord[np.r_[7262:7275,7356:7368]] =np.array([-1,0])
location_coord[np.r_[5068:5129]] =np.array([-5,11])
location_coord[np.r_[5130:5155]] =np.array([-5,10])
location_coord[np.r_[5231:5260]] =np.array([-5,8])
location_coord[np.r_[5261:5325,5974:6065]] =np.array([-5,7])
location_coord[np.r_[6142:6178]] =np.array([-5,6])
location_coord[np.r_[6179:6225]] =np.array([-5,5])
location_coord[np.r_[1340:1437]] =np.array([-5,4])
location_coord[np.r_[1009:1040]] =np.array([-5,3])
location_coord[np.r_[991:1008]] =np.array([-5,2])
location_coord[np.r_[970:990]] =np.array([-5,1])
location_coord[np.r_[930:969]] =np.array([-5,0])
location_coord[np.r_[5046:5067]] =np.array([-4,11])
location_coord[np.r_[5156:5187]] =np.array([-4,10])
location_coord[np.r_[5188:5200]] =np.array([-4,9])
location_coord[np.r_[5201:5230]] =np.array([-4,8])
location_coord[np.r_[5326:5345,5942:5973,6066:6106]] =np.array([-4,7])
location_coord[np.r_[6107:6141]] =np.array([-4,6])
location_coord[np.r_[6226:6305]] =np.array([-4,5])
location_coord[np.r_[1326:1339,1438:1464,6306:6354]] =np.array([-4,4])
location_coord[np.r_[1041:1110]] =np.array([-4,3])
location_coord[np.r_[1111:1166]] =np.array([-4,2])
location_coord[np.r_[846:887]] =np.array([-4,1])
location_coord[np.r_[888:929]] =np.array([-4,0])
location_coord[np.r_[5000:5045]] =np.array([-3,11])
location_coord[np.r_[4778:4823,4945:4999]] =np.array([-3,10])

location_coord[np.r_[5362:5382]] =np.array([-3,8])

location_coord[np.r_[5346:5361,5885:5941]] =np.array([-3,7])

location_coord[np.r_[5839:5884]] =np.array([-3,6])

location_coord[np.r_[6355:6372]] =np.array([-3,4])

location_coord[np.r_[1269:1297]] =np.array([-3,3])

location_coord[np.r_[1167:1185]] =np.array([-3,2])

location_coord[np.r_[800:845]] =np.array([-3,1])

location_coord[np.r_[745:799]] =np.array([-3,0])

location_coord[np.r_[4702:4719]] =np.array([-2,11])

location_coord[np.r_[4824:4836,4932:4944]] =np.array([-2,10])

location_coord[np.r_[5427:5476]] =np.array([-2,9])

location_coord[np.r_[5383:5426]] =np.array([-2,8])

location_coord[np.r_[5720:5805,6479:6565]] =np.array([-2,7])

location_coord[np.r_[5806:5838,6460:6478]] =np.array([-2,6])

location_coord[np.r_[6439:6459]] =np.array([-2,5])

location_coord[np.r_[6373:6438]] =np.array([-2,4])

location_coord[np.r_[1236:1268]] =np.array([-2,3])

location_coord[np.r_[1186:1235]] =np.array([-2,2])

location_coord[np.r_[653:705]] =np.array([-2,1])

location_coord[np.r_[706:744,7276:7355]] =np.array([-2,0])

location_coord[np.r_[4683:4701]] =np.array([-1,11])

location_coord[np.r_[4837:4931]] =np.array([-1,10])

location_coord[np.r_[5744:5515]] =np.array([-1,9])

location_coord[np.r_[5516:5537]] =np.array([-1,8])

location_coord[np.r_[5538:5572,5648:5719,6566:6596]] =np.array([-1,7])

location_coord[np.r_[6597:6615]] =np.array([-1,6])

location_coord[np.r_[1521:1531,6616:6642]] =np.array([-1,5])




# x_user = int(input("Enter the x-coordinate : "))



location_coord[np.r_[1507:1520]] =np.array([-1,4])

location_coord[np.r_[631:652]] =np.array([-1,1])

location_coord[np.r_[7262:7275,7356:7368]] =np.array([-1,0])

location_coord[np.r_[4405:4445,4670:4682]] =np.array([0,11])

location_coord[np.r_[4389:4404]] =np.array([0,10])

location_coord[np.r_[4373:4388]] =np.array([0,9])

location_coord[np.r_[4323:4372]] =np.array([0,8])

location_coord[np.r_[5573:5647]] =np.array([0,7])

location_coord[np.r_[1561:1605,6686:6734]] =np.array([0,6])

location_coord[np.r_[1532:1560,6643:6685]] =np.array([0,5])

location_coord[np.r_[409:462]] =np.array([0,3])

location_coord[np.r_[463:508]] =np.array([0,2])

location_coord[np.r_[609:630]] =np.array([0,1])

location_coord[np.r_[0:46,7247:7261,7369:7451]] =np.array([0,0])

location_coord[np.r_[4446:4500,4621:4669]] =np.array([1,11])

location_coord[np.r_[4501:4512,4607:4620]] =np.array([1,10])

location_coord[np.r_[4513:4606]] =np.array([1,9])

location_coord[np.r_[4276:4322]] =np.array([1,8])

location_coord[np.r_[4263:4275]] =np.array([1,7])

location_coord[np.r_[1606:1647,4249:4262,6735:6772]] =np.array([1,6])

location_coord[np.r_[1648:1662,4206:4248,6773:6838]] =np.array([1,5])

location_coord[np.r_[1663:1720,6839:6875]] =np.array([1,4])

location_coord[np.r_[358:408]] =np.array([1,3])

location_coord[np.r_[509:524]] =np.array([1,2])

location_coord[np.r_[598:608]] =np.array([1,1])

location_coord[np.r_[47:59,7234:7426]] =np.array([1,0])

location_coord[np.r_[3552:3618]] =np.array([2,11])

location_coord[np.r_[3619:3652]] =np.array([2,10])

location_coord[np.r_[3778:3904]] =np.array([2,9])

location_coord[np.r_[3905:3955]] =np.array([2,8])

location_coord[np.r_[4054:4124]] =np.array([2,7])

location_coord[np.r_[4215:4138]] =np.array([2,6])

location_coord[np.r_[4139:4205]] =np.array([2,5])

location_coord[np.r_[293:332,1721:1740,6876:6885]] =np.array([2,4])

location_coord[np.r_[225:292]] =np.array([2,3])

location_coord[np.r_[525:560]] =np.array([2,2])

location_coord[np.r_[561:597]] =np.array([2,1])

location_coord[np.r_[60:69,7209:7233]] =np.array([2,0])

location_coord[np.r_[3529:3551]] =np.array([3,11])

location_coord[np.r_[3653:3671]] =np.array([3,10])

location_coord[np.r_[3756:3777]] =np.array([3,9])

location_coord[np.r_[3956:4013]] =np.array([3,8])

location_coord[np.r_[4014:4053]] =np.array([3,7])

location_coord[np.r_[6987:7079]] =np.array([3,6])

location_coord[np.r_[6960:6986,7080:7095]] =np.array([3,5])


location_coord[np.r_[190:224,1741:1753,6886:6959,7096:7119]] =np.array([3,4])
location_coord[np.r_[7120:7143]] =np.array([3,3])
location_coord[np.r_[132:150,2980:3021,7144:7160]] =np.array([3,2])
location_coord[np.r_[122:131,2943:2979,7161:7172]] =np.array([3,1])
location_coord[np.r_[70:121,7173:7208]] =np.array([3,0])
location_coord[np.r_[3506:3528]] =np.array([4,11])
location_coord[np.r_[3672:3726]] =np.array([4,10])
location_coord[np.r_[3727:3755]] =np.array([4,9])

location_coord[np.r_[1846:1866,3104:3132]] =np.array([4,7])
location_coord[np.r_[1831:1845,3091:3103]] =np.array([4,6])
location_coord[np.r_[1812:1830,3076:3090]] =np.array([4,5])
location_coord[np.r_[1754:1811,3042:3075]] =np.array([4,4])

location_coord[np.r_[2663:2710]] =np.array([4,2])
location_coord[np.r_[2711:2723,2933:2942]] =np.array([4,1])
location_coord[np.r_[2724:2769,2889:2932]] =np.array([4,0])
location_coord[np.r_[3464:3505]] =np.array([5,11])
location_coord[np.r_[3304:3352]] =np.array([5,10])
location_coord[np.r_[1881:1907,3196:3227]] =np.array([5,9])
location_coord[np.r_[1867:1880,3133:3195]] =np.array([5,8])

location_coord[np.r_[2055:2109]] =np.array([5,6])
location_coord[np.r_[2110:2125]] =np.array([5,5])
location_coord[np.r_[2126:2140]] =np.array([5,4])
location_coord[np.r_[2141:2184]] =np.array([5,3])
location_coord[np.r_[2403:2447,2624:2662]] =np.array([5,2])
location_coord[np.r_[2448:2490,2574:2623]] =np.array([5,1])
location_coord[np.r_[2770:2785,2865:2888]] =np.array([5,0])
location_coord[np.r_[3353:3463]] =np.array([6,11])
location_coord[np.r_[3263:3303]] =np.array([6,10])
location_coord[np.r_[1908:1970,3228:3262]] =np.array([6,9])
location_coord[np.r_[1971:1985]] =np.array([6,8])
location_coord[np.r_[1986:1998]] =np.array([6,7])
location_coord[np.r_[1999:2054]] =np.array([6,6])

location_coord[np.r_[2237:2335]] =np.array([6,5])
location_coord[np.r_[2224:2236,2336:2345]] =np.array([6,4])
location_coord[np.r_[2185:2223,2346:2358]] =np.array([6,3])
location_coord[np.r_[2359:2402]] =np.array([6,2])
location_coord[np.r_[2491:2573]] =np.array([6,1])
location_coord[np.r_[2786:2864]] =np.array([6,0])

def transform_and_normalize(index):
    #if not (-5 <= x_original <= 6) or not (0 <= y_original <= 11):
     #   raise ValueError("Coordinates must satisfy -5 <= x <= 6 and 0 <= y <= 11.")
    # print(index)
    # index=int(index)
    x_original = location_coord[index][0]
    y_original = location_coord[index][1]
    
    x_transformed = x_original + 5
    y_transformed = 11 - y_original


    x_normalized = x_transformed * (309//11.5)
    y_normalized = y_transformed * (275//11.8)
    if x_normalized == 0 : x_normalized += 5
    if y_normalized == 0 : y_normalized += 5
    # print(x_original,y_original)
    return int(x_normalized), int(y_normalized)
# y_user = int(input("Enter the y-coordinate : "))
index = int(input("Enter the required index : "))

normalized_coordinates = transform_and_normalize(index)
print("Normalized Coordinates:", normalized_coordinates)
# print(location_coord.shape[0])

for i in range(location_coord.shape[0]):
    # print(i)
    location_coord[i] = transform_and_normalize(i)
# print(np.isnan(location_coord).sum())
# # # print(location_coord[1:])
# location_coord[:][:,0]= (location_coord[:][0]+5)*(309//11.5)
# location_coord[:][1]= (-location_coord[:][1]+11)*(275//11.5)
# # location_coord=np.where(location_coord==0,5,location_coord)
# n_location_coord = np.array([transform_and_normalize(i) for i in range(location_coord.shape[0])])
print(location_coord[50])
# print(location_coord)
with open("location_coord.pkl", "wb") as f:
    pickle.dump(location_coord, f)

