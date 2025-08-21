//
// Created by kocur on 09-May-25.
//

#ifndef POSELIB_RELPOSE_3V_TRF_H
#define POSELIB_RELPOSE_3V_TRF_H

#include "PoseLib/types.h"
#include "PoseLib/camera_pose.h"

namespace poselib {
void relpose_3v_trf(const std::vector <Point2D> &x1, const std::vector <Point2D> &x2, const std::vector <Point2D> &x3,
                   double alpha, const RelativePoseOptions &opt, std::vector <ImageTriplet> *models);

}


#endif //POSELIB_RELPOSE_3V_TRF_H
