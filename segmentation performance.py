from statistics import harmonic_mean


def calArea (rec : list):
    area = max(0, (rec[2] - rec[0])) * max(0, (rec[3] - rec[1]))

    return area

def overlapping (user : list, computer : list):
    covered  = []

    if (user[0] > computer[0]):
        x = user[0]
    else:
        x = computer[0]
    
    if (user[1] > computer[1]):
        y = user[1]
    else:
        y = computer[1]

    if (user[2] > computer[2]):
        z = computer[2]
    else:
        z = user[2]

    if (user[3] > computer[3]):
        a = computer[3]
    else:
        a = user[3]

    covered = [x, y, z, a]    

    return covered


def IoU (user : list, computer: list):
    overlap = overlapping(user, computer)
    areaOverlap = calArea(overlap)

    areaUser = calArea(user)
    areaComputer = calArea(computer)

    print(areaUser, areaComputer, areaOverlap)

    areaUnion = areaUser + areaComputer - areaOverlap

    if areaUnion == 0:
        return 0
    
    idx = areaOverlap / areaUnion    

    return idx

# test = IoU([50, 50, 150, 150], [100, 100, 200, 200])
# print(test)




def performance(user : list, computer : list) :
    threshold = float(input("Please enter a threshold value: "))

    try:
        index = float(threshold)
        print("You have entered the threshold value:", index)
    except ValueError:
        print("Invalid input. Please enter a numeric value.")
    
    
    match = 0

    for i in range (len(computer)) :

        for j in range (len(user)):
            intersection = IoU(user[j], computer[i])
            print(computer[i], user[j], intersection)
            if (intersection > threshold):
                match = match + 1

    precision = match / (len(computer))
    recall = match / (len(user))
    fscore = harmonic_mean([precision, recall])

    print("The precision is ", precision)
    print("The recall is ", recall)
    print("The f-score is ", fscore)

# Example usage
ground_truth = [(50, 50, 100, 100), (150, 150, 200, 200)]
detected = [(55, 55, 95, 95), (160, 160, 190, 190), (300, 300, 350, 350)]

performance([(50, 50, 100, 100), (150, 150, 200, 200)], [(55, 55, 95, 95), (160, 160, 190, 190), (300, 300, 350, 350)])