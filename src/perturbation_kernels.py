import numpy as np

def cosine_phi_12(k_1, k_2, k_3):
    return 0.5*(k_3*k_3-k_1*k_1-k_2*k_2)/(k_1*k_2) # law of cosines (with a negative sign to get the complementary angle outside the triangle)
    
def F2_EdS_angular(k_1, k_2, cos_phi_12):
    return 5./7. + 0.5*cos_phi_12*(k_1/k_2 + k_2/k_1) + 2./7.*cos_phi_12*cos_phi_12

def F2_EdS(k_1, k_2, k_3):
    cos_phi_12 = 0.5*(k_3*k_3-k_1*k_1-k_2*k_2)/(k_1*k_2) # law of cosines (with a negative sign to get the complementary angle outside the triangle)
    return 5./7. + 0.5*cos_phi_12*(k_1/k_2 + k_2/k_1) + 2./7.*cos_phi_12*cos_phi_12

def S2_angular(cos_phi_12):
    return cos_phi_12*cos_phi_12 - 1./3.
    
def S2(k_1, k_2, k_3):
    cos_phi_12 = 0.5*(k_3*k_3-k_1*k_1-k_2*k_2)/(k_1*k_2) # law of cosines (with a negative sign to get the complementary angle outside the triangle)
    return cos_phi_12*cos_phi_12 - 1./3.

"""

## the first 3 components of the q_vec should be the Cartesian coordinates and the 4th component should be the vector's length i.e. [q_x,q_y,q_z,q]

## m stands for minus (-) and p stands for plus (+)

std::vector<double> q_mA_vec(const std::vector<double> &q_A_vec)
{
    return std::vector<double> {-q_A_vec.at(0), -q_A_vec.at(1), -q_A_vec.at(2), q_A_vec.at(3)};
}

std::vector<double> q_mA_perturbed_vec(const std::vector<double> &q_A_vec)
{
    double pert_factor = 0.99999;

    double x = -q_A_vec.at(0)*pert_factor;
    double y = -q_A_vec.at(1)*pert_factor;
    double z = -q_A_vec.at(2)*pert_factor;

    double r = sqrt(x*x + y*y + z*z);
    return std::vector<double> {x, y, z, r};
}

std::vector<double> q_ApB_vec(const std::vector<double> &q_A_vec, const std::vector<double> &q_B_vec)
{
    double q_ApB_x = q_A_vec.at(0) + q_B_vec.at(0);
    double q_ApB_y = q_A_vec.at(1) + q_B_vec.at(1);
    double q_ApB_z = q_A_vec.at(2) + q_B_vec.at(2);
    double q_ApB = sqrt(q_ApB_x*q_ApB_x + q_ApB_y*q_ApB_y + q_ApB_z*q_ApB_z);
    return std::vector<double> {q_ApB_x, q_ApB_y, q_ApB_z, q_ApB};
}

double cos_phi_AB(const std::vector<double> &q_A_vec, const std::vector<double> &q_B_vec)
{
    return (q_A_vec.at(0)*q_B_vec.at(0) + q_A_vec.at(1)*q_B_vec.at(1) + q_A_vec.at(2)*q_B_vec.at(2) ) / (q_A_vec.at(3)*q_B_vec.at(3));
}

bool is_vec_alright(const std::vector<double> &q_A_vec)
{
    if (isnan(q_A_vec.at(0)) || isnan(q_A_vec.at(1)) || isnan(q_A_vec.at(2)))
        return false;

    else
        return true;
}

// ######################################################################################

double alpha_EdS(const double &k_1, const double &k_2, const double &cos_phi_12)
{
    if(k_1 == 0)
        return 0;

    return 1.0 + k_2/k_1*cos_phi_12;
}

double beta_EdS(const double &k_1, const double &k_2, const double &cos_phi_12)
{
    if(k_1 == 0 || k_2 == 0)
        return 0;

    return 0.5*cos_phi_12*(k_1/k_2 + k_2/k_1 + 2*cos_phi_12);
}

double F2(const double &q_1, const double &q_2, const double &cos_phi_12)
{
    double alpha_1_2 = alpha_EdS(q_1,q_2,cos_phi_12);
    double beta_1_2 = beta_EdS(q_1,q_2,cos_phi_12);
    return 5/7.0*alpha_1_2 + 2/7.0*beta_1_2;
}

double G2(const double &q_1, const double &q_2, const double &cos_phi_12)
{
    double alpha_1_2 = alpha_EdS(q_1,q_2,cos_phi_12);
    double beta_1_2 = beta_EdS(q_1,q_2,cos_phi_12);
    return 3/7.0*alpha_1_2 + 4/7.0*beta_1_2;
}

double F2(const std::vector<double> &q_1_vec, const std::vector<double> &q_2_vec)
{
    double cos_phi_1_2 = cos_phi_AB(q_1_vec, q_2_vec);

    double alpha_1_2 = alpha_EdS(q_1_vec.at(3),q_2_vec.at(3),cos_phi_1_2);
    double beta_1_2 = beta_EdS(q_1_vec.at(3),q_2_vec.at(3),cos_phi_1_2);
    return 5/7.0*alpha_1_2 + 2/7.0*beta_1_2;
}

double G2(const std::vector<double> &q_1_vec, const std::vector<double> &q_2_vec)
{
    double cos_phi_1_2 = cos_phi_AB(q_1_vec, q_2_vec);

    double alpha_1_2 = alpha_EdS(q_1_vec.at(3),q_2_vec.at(3),cos_phi_1_2);
    double beta_1_2 = beta_EdS(q_1_vec.at(3),q_2_vec.at(3),cos_phi_1_2);
    return 3/7.0*alpha_1_2 + 4/7.0*beta_1_2;
}

double F3(const std::vector<double> &q_1_vec, const std::vector<double> &q_2_vec, const std::vector<double> &q_3_vec)
{
    std::vector<double> q_1p2_vec = q_ApB_vec(q_1_vec, q_2_vec);
    std::vector<double> q_2p3_vec = q_ApB_vec(q_2_vec, q_3_vec);

    double cos_phi_1_2p3 = cos_phi_AB(q_1_vec, q_2p3_vec);
    double alpha_1_2p3 = alpha_EdS(q_1_vec.at(3),q_2p3_vec.at(3),cos_phi_1_2p3);
    double beta_1_2p3 = beta_EdS(q_1_vec.at(3),q_2p3_vec.at(3),cos_phi_1_2p3);

    double cos_phi_1p2_3 = cos_phi_AB(q_1p2_vec, q_3_vec);
    double alpha_1p2_3 = alpha_EdS(q_1p2_vec.at(3),q_3_vec.at(3),cos_phi_1p2_3);
    double beta_1p2_3 = beta_EdS(q_1p2_vec.at(3),q_3_vec.at(3),cos_phi_1p2_3);

    double F2_2_3 = F2(q_2_vec,q_3_vec);
    double G2_1_2 = G2(q_1_vec,q_2_vec);
    double G2_2_3 = G2(q_2_vec,q_3_vec);

    return 7/18.0*alpha_1_2p3*F2_2_3 + 2/18.0*beta_1_2p3*G2_2_3 + 7/18.0*alpha_1p2_3*G2_1_2 + 2/18.0*beta_1p2_3*G2_1_2;
}

double G3(const std::vector<double> &q_1_vec, const std::vector<double> &q_2_vec, const std::vector<double> &q_3_vec)
{
    std::vector<double> q_2p3_vec = q_ApB_vec(q_2_vec, q_3_vec);
    std::vector<double> q_1p2_vec = q_ApB_vec(q_1_vec, q_2_vec);

    double cos_phi_1_2p3 = cos_phi_AB(q_1_vec, q_2p3_vec);
    double alpha_1_2p3 = alpha_EdS(q_1_vec.at(3),q_2p3_vec.at(3),cos_phi_1_2p3);
    double beta_1_2p3 = beta_EdS(q_1_vec.at(3),q_2p3_vec.at(3),cos_phi_1_2p3);

    double cos_phi_1p2_3 = cos_phi_AB(q_1p2_vec, q_3_vec);
    double alpha_1p2_3 = alpha_EdS(q_1p2_vec.at(3),q_3_vec.at(3),cos_phi_1p2_3);
    double beta_1p2_3 = beta_EdS(q_1p2_vec.at(3),q_3_vec.at(3),cos_phi_1p2_3);

    double F2_2_3 = F2(q_2_vec,q_3_vec);
    double G2_1_2 = G2(q_1_vec,q_2_vec);
    double G2_2_3 = G2(q_2_vec,q_3_vec);

    return 3/18.0*alpha_1_2p3*F2_2_3 + 6/18.0*beta_1_2p3*G2_2_3 + 3/18.0*alpha_1p2_3*G2_1_2 + 6/18.0*beta_1p2_3*G2_1_2;
}

double F4(const std::vector<double> &q_1_vec, const std::vector<double> &q_2_vec, const std::vector<double> &q_3_vec, const std::vector<double> &q_4_vec)
{
    std::vector<double> q_2p3p4_vec = q_ApB_vec(q_ApB_vec(q_2_vec, q_3_vec),q_4_vec);
    std::vector<double> q_1p2_vec = q_ApB_vec(q_1_vec, q_2_vec);
    std::vector<double> q_3p4_vec = q_ApB_vec(q_3_vec, q_4_vec);
    std::vector<double> q_1p2p3_vec = q_ApB_vec(q_ApB_vec(q_1_vec, q_2_vec),q_3_vec);

    double cos_phi_1_2p3p4 = cos_phi_AB(q_1_vec, q_2p3p4_vec);
    double alpha_1_2p3p4 = alpha_EdS(q_1_vec.at(3),q_2p3p4_vec.at(3),cos_phi_1_2p3p4);
    double beta_1_2p3p4 = beta_EdS(q_1_vec.at(3),q_2p3p4_vec.at(3),cos_phi_1_2p3p4);

    double cos_phi_1p2_3p4 = cos_phi_AB(q_1p2_vec, q_3p4_vec);
    double alpha_1p2_3p4 = alpha_EdS(q_1p2_vec.at(3),q_3p4_vec.at(3),cos_phi_1p2_3p4);
    double beta_1p2_3p4 = beta_EdS(q_1p2_vec.at(3),q_3p4_vec.at(3),cos_phi_1p2_3p4);

    double cos_phi_1p2p3_4 = cos_phi_AB(q_1p2p3_vec, q_4_vec);
    double alpha_1p2p3_4 = alpha_EdS(q_1p2p3_vec.at(3),q_4_vec.at(3),cos_phi_1p2p3_4);
    double beta_1p2p3_4 = beta_EdS(q_1p2p3_vec.at(3),q_4_vec.at(3),cos_phi_1p2p3_4);

    double F3_2_3_4 = F3(q_2_vec,q_3_vec,q_4_vec);
    double G3_2_3_4 = G3(q_2_vec,q_3_vec,q_4_vec);

    double G2_1_2 = G2(q_1_vec,q_2_vec);
    double F2_3_4 = F2(q_3_vec,q_4_vec);
    double G2_3_4 = G2(q_3_vec,q_4_vec);

    double G3_1_2_3 = G3(q_1_vec,q_2_vec,q_3_vec);

    return 9/33.0*alpha_1_2p3p4*F3_2_3_4 + 2/33.0*beta_1_2p3p4*G3_2_3_4 + 9/33.0*alpha_1p2_3p4*G2_1_2*F2_3_4 + 2/33.0*beta_1p2_3p4*G2_1_2*G2_3_4
            + 9/33.0*alpha_1p2p3_4*G3_1_2_3 + 2/33.0*beta_1p2p3_4*G3_1_2_3;
}

double G4(const std::vector<double> &q_1_vec, const std::vector<double> &q_2_vec, const std::vector<double> &q_3_vec, const std::vector<double> &q_4_vec)
{
    std::vector<double> q_2p3p4_vec = q_ApB_vec(q_ApB_vec(q_2_vec, q_3_vec),q_4_vec);
    std::vector<double> q_1p2_vec = q_ApB_vec(q_1_vec, q_2_vec);
    std::vector<double> q_3p4_vec = q_ApB_vec(q_3_vec, q_4_vec);
    std::vector<double> q_1p2p3_vec = q_ApB_vec(q_ApB_vec(q_1_vec, q_2_vec),q_3_vec);

    double cos_phi_1_2p3p4 = cos_phi_AB(q_1_vec, q_2p3p4_vec);
    double alpha_1_2p3p4 = alpha_EdS(q_1_vec.at(3),q_2p3p4_vec.at(3),cos_phi_1_2p3p4);
    double beta_1_2p3p4 = beta_EdS(q_1_vec.at(3),q_2p3p4_vec.at(3),cos_phi_1_2p3p4);

    double cos_phi_1p2_3p4 = cos_phi_AB(q_1p2_vec, q_3p4_vec);
    double alpha_1p2_3p4 = alpha_EdS(q_1p2_vec.at(3),q_3p4_vec.at(3),cos_phi_1p2_3p4);
    double beta_1p2_3p4 = beta_EdS(q_1p2_vec.at(3),q_3p4_vec.at(3),cos_phi_1p2_3p4);

    double cos_phi_1p2p3_4 = cos_phi_AB(q_1p2p3_vec, q_4_vec);
    double alpha_1p2p3_4 = alpha_EdS(q_1p2p3_vec.at(3),q_4_vec.at(3),cos_phi_1p2p3_4);
    double beta_1p2p3_4 = beta_EdS(q_1p2p3_vec.at(3),q_4_vec.at(3),cos_phi_1p2p3_4);

    double F3_2_3_4 = F3(q_2_vec,q_3_vec,q_4_vec);
    double G3_2_3_4 = G3(q_2_vec,q_3_vec,q_4_vec);

    double G2_1_2 = G2(q_1_vec,q_2_vec);
    double F2_3_4 = F2(q_3_vec,q_4_vec);
    double G2_3_4 = G2(q_3_vec,q_4_vec);

    double G3_1_2_3 = G3(q_1_vec,q_2_vec,q_3_vec);

    return 3/33.0*alpha_1_2p3p4*F3_2_3_4 + 8/33.0*beta_1_2p3p4*G3_2_3_4 + 3/33.0*alpha_1p2_3p4*G2_1_2*F2_3_4 + 8/33.0*beta_1p2_3p4*G2_1_2*G2_3_4
            + 3/33.0*alpha_1p2p3_4*G3_1_2_3 + 8/33.0*beta_1p2p3_4*G3_1_2_3;
}

// symmetrized kernels

double F2_sym(const std::vector<double> &q_1_vec, const std::vector<double> &q_2_vec)
{
    if(q_1_vec.at(3) == 0 || q_2_vec.at(3) == 0)
        return 0;

    return 1/2.0*(F2(q_1_vec,q_2_vec)+F2(q_2_vec,q_1_vec));
}

double G2_sym(const std::vector<double> &q_1_vec, const std::vector<double> &q_2_vec)
{
    if(q_1_vec.at(3) == 0 || q_2_vec.at(3) == 0)
        return 0;

    return 1/2.0*(G2(q_1_vec,q_2_vec)+G2(q_2_vec,q_1_vec));
}

double F3_sym(const std::vector<double> &q_1_vec, const std::vector<double> &q_2_vec, const std::vector<double> &q_3_vec)
{
    if(q_1_vec.at(3) == 0 || q_2_vec.at(3) == 0 || q_3_vec.at(3) == 0)
        return 0;

    return 1/6.0*(F3(q_1_vec,q_2_vec,q_3_vec)+F3(q_2_vec,q_3_vec,q_1_vec)+F3(q_3_vec,q_1_vec,q_2_vec)+
                  F3(q_1_vec,q_3_vec,q_2_vec)+F3(q_2_vec,q_1_vec,q_3_vec)+F3(q_3_vec,q_2_vec,q_1_vec));
}

double F3_sym(const double &q_1, const double &q_2, const double &q_3, const double &mu_12, const double &mu_13, const double &mu_23)
{
    if(q_1 == 0 || q_3 == 0 || q_3 == 0)
        return 0;

    std::vector<double> q_1_vec = {q_1, 0, 0, q_1};
    std::vector<double> q_2_vec = {q_2*mu_12, q_2*sqrt(1 - mu_12*mu_12), 0, q_2};
    double q_3_x = q_3*mu_13;
    double q_3_y = (q_2*q_3*mu_23 - q_3_x*q_2_vec.at(0)) / q_2_vec.at(1);
    double q_3_z = sqrt(q_3*q_3 - q_3_x*q_3_x - q_3_y*q_3_y);
    std::vector<double> q_3_vec = {q_3_x, q_3_y, q_3_z, q_3};

    return F3_sym(q_1_vec,q_2_vec,q_3_vec);
}

double G3_sym(const std::vector<double> &q_1_vec, const std::vector<double> &q_2_vec, const std::vector<double> &q_3_vec)
{
    if(q_1_vec.at(3) == 0 || q_2_vec.at(3) == 0 || q_3_vec.at(3) == 0)
        return 0;

    return 1/6.0*(G3(q_1_vec,q_2_vec,q_3_vec)+G3(q_2_vec,q_3_vec,q_1_vec)+G3(q_3_vec,q_1_vec,q_2_vec)+
                  G3(q_1_vec,q_3_vec,q_2_vec)+G3(q_2_vec,q_1_vec,q_3_vec)+G3(q_3_vec,q_2_vec,q_1_vec));
}

double F4_sym(const std::vector<double> &q_1_vec, const std::vector<double> &q_2_vec, const std::vector<double> &q_3_vec, const std::vector<double> &q_4_vec)
{
    if(q_1_vec.at(3) == 0 || q_2_vec.at(3) == 0 || q_3_vec.at(3) == 0 || q_4_vec.at(3) == 0)
        return 0;

    return 1/24.0*(F4(q_1_vec,q_2_vec,q_3_vec,q_4_vec)+F4(q_2_vec,q_3_vec,q_4_vec,q_1_vec)+F4(q_3_vec,q_4_vec,q_1_vec,q_2_vec)+F4(q_4_vec,q_1_vec,q_2_vec,q_3_vec)+
                   F4(q_1_vec,q_3_vec,q_4_vec,q_2_vec)+F4(q_2_vec,q_4_vec,q_1_vec,q_3_vec)+F4(q_3_vec,q_1_vec,q_2_vec,q_4_vec)+F4(q_4_vec,q_2_vec,q_3_vec,q_1_vec)+
                   F4(q_1_vec,q_4_vec,q_2_vec,q_3_vec)+F4(q_2_vec,q_1_vec,q_3_vec,q_4_vec)+F4(q_3_vec,q_2_vec,q_4_vec,q_1_vec)+F4(q_4_vec,q_3_vec,q_1_vec,q_2_vec)+
                   F4(q_1_vec,q_2_vec,q_4_vec,q_3_vec)+F4(q_2_vec,q_3_vec,q_1_vec,q_4_vec)+F4(q_3_vec,q_4_vec,q_2_vec,q_1_vec)+F4(q_4_vec,q_1_vec,q_3_vec,q_2_vec)+
                   F4(q_1_vec,q_3_vec,q_2_vec,q_4_vec)+F4(q_2_vec,q_4_vec,q_3_vec,q_1_vec)+F4(q_3_vec,q_1_vec,q_4_vec,q_2_vec)+F4(q_4_vec,q_2_vec,q_1_vec,q_3_vec)+
                   F4(q_1_vec,q_4_vec,q_3_vec,q_2_vec)+F4(q_2_vec,q_1_vec,q_4_vec,q_3_vec)+F4(q_3_vec,q_2_vec,q_1_vec,q_4_vec)+F4(q_4_vec,q_3_vec,q_2_vec,q_1_vec));
}

double F4_sym(const double &q_1, const double &q_2, const double &q_3, const double &q_4,
              const double &mu_12, const double &mu_13, const double &mu_14, const double &mu_23, const double &mu_24, const double &mu_34)
{
    if(q_1 == 0 || q_2 == 0 || q_3 == 0 || q_4 == 0)
        return 0;

    std::vector<double> q_1_vec = {q_1, 0, 0, q_1};
    std::vector<double> q_2_vec = {q_2*mu_12, q_2*sqrt(1 - mu_12*mu_12), 0, q_2};
    double q_3_x = q_3*mu_13;
    double q_3_y = (q_2*q_3*mu_23 - q_3_x*q_2_vec.at(0)) / q_2_vec.at(1);
    double q_3_z = sqrt(q_3*q_3 - q_3_x*q_3_x - q_3_y*q_3_y);
    std::vector<double> q_3_vec = {q_3_x, q_3_y, q_3_z, q_3};

    double q_4_x = q_4*mu_14;
    double q_4_y = (q_2*q_4*mu_24 - q_4_x*q_2_vec.at(0)) / q_2_vec.at(1);
    double q_4_z = (q_3*q_4*mu_34 - q_4_x*q_3_vec.at(0) - q_4_y*q_3_vec.at(1)) / q_3_vec.at(2);

    double q_4_computed = sqrt(q_4_x*q_4_x + q_4_y*q_4_y + q_4_z*q_4_z);
    if (q_4_computed != q_4)
    {
        std::cout << q_4_computed << " is not equal to the length provided i.e. " << q_4 << std::endl;
        //assert (q_4_computed == q_4);
    }

    std::vector<double> q_4_vec = {q_4_x, q_4_y, q_4_z, q_4_computed};

    std::cout << "\n" << q_1_vec.at(0) << " " << q_1_vec.at(1) << " " << q_1_vec.at(2) << " " << q_1_vec.at(3) << std::endl;
    std::cout << q_2_vec.at(0) << " " << q_2_vec.at(1) << " " << q_2_vec.at(2) << " " << q_2_vec.at(3) << std::endl;
    std::cout << q_3_vec.at(0) << " " << q_3_vec.at(1) << " " << q_3_vec.at(2) << " " << q_3_vec.at(3) << std::endl;
    std::cout << q_4_vec.at(0) << " " << q_4_vec.at(1) << " " << q_4_vec.at(2) << " " << q_4_vec.at(3) << std::endl;

    return F4_sym(q_1_vec,q_2_vec,q_3_vec,q_4_vec);
}

double G4_sym(const std::vector<double> &q_1_vec, const std::vector<double> &q_2_vec, const std::vector<double> &q_3_vec, const std::vector<double> &q_4_vec)
{
    if(q_1_vec.at(3) == 0 || q_2_vec.at(3) == 0 || q_3_vec.at(3) == 0 || q_4_vec.at(3) == 0)
        return 0;

    return 1/24.0*(G4(q_1_vec,q_2_vec,q_3_vec,q_4_vec)+G4(q_2_vec,q_3_vec,q_4_vec,q_1_vec)+G4(q_3_vec,q_4_vec,q_1_vec,q_2_vec)+G4(q_4_vec,q_1_vec,q_2_vec,q_3_vec)+
                   G4(q_1_vec,q_3_vec,q_4_vec,q_2_vec)+G4(q_2_vec,q_4_vec,q_1_vec,q_3_vec)+G4(q_3_vec,q_1_vec,q_2_vec,q_4_vec)+G4(q_4_vec,q_2_vec,q_3_vec,q_1_vec)+
                   G4(q_1_vec,q_4_vec,q_2_vec,q_3_vec)+G4(q_2_vec,q_1_vec,q_3_vec,q_4_vec)+G4(q_3_vec,q_2_vec,q_4_vec,q_1_vec)+G4(q_4_vec,q_3_vec,q_1_vec,q_2_vec)+
                   G4(q_1_vec,q_2_vec,q_4_vec,q_3_vec)+G4(q_2_vec,q_3_vec,q_1_vec,q_4_vec)+G4(q_3_vec,q_4_vec,q_2_vec,q_1_vec)+G4(q_4_vec,q_1_vec,q_3_vec,q_2_vec)+
                   G4(q_1_vec,q_3_vec,q_2_vec,q_4_vec)+G4(q_2_vec,q_4_vec,q_3_vec,q_1_vec)+G4(q_3_vec,q_1_vec,q_4_vec,q_2_vec)+G4(q_4_vec,q_2_vec,q_1_vec,q_3_vec)+
                   G4(q_1_vec,q_4_vec,q_3_vec,q_2_vec)+G4(q_2_vec,q_1_vec,q_4_vec,q_3_vec)+G4(q_3_vec,q_2_vec,q_1_vec,q_4_vec)+G4(q_4_vec,q_3_vec,q_2_vec,q_1_vec));
}
"""