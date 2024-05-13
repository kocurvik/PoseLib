#include <Eigen/Dense>

#include "PoseLib/camera_pose.h"
#include "PoseLib/solvers/rd_utils.h"
#include "PoseLib/misc/sturm.h"
#include "PoseLib/misc/essential.h"
#include "PoseLib/solvers/relpose_kfEfk_utils.h"

namespace poselib {
    //#define MAXIT 30 // maximal number of iterations
#define MAXIT 50 // maximal number of iterations
#define MAXTREE (2*MAXIT + 1) // maximal depth of the isolating tree
//#define ACC 1.0e-12 // root is polished to an approximate accuracy ACC
#define ACC 1.0e-16 // root is polished to an approximate accuracy ACC
// #define DEG MAXSOLS // degree of the polynomial


    template<typename Derived>
    void charpoly_krylov(Eigen::MatrixBase<Derived> &A, double *p) {
        int n = A.rows();
        Eigen::MatrixXd V(n, n);

        // TODO: Figure out what a good choice is here
        V.col(0) = Eigen::VectorXd::Ones(n);
        V(n - 1, 0) = 0;

        for (int i = 1; i < n; i++)
            V.col(i) = A * V.col(i - 1);

        Eigen::VectorXd c(n);
        c = V.partialPivLu().solve(A*V.col(n - 1));

        p[n] = 1;
        for (int i = 0; i < n; i++)
            p[i] = -c(i);
    }

/* Computes characteristic poly using Danilevsky's method. */
    template<typename Derived>
    void charpoly_danilevsky(Eigen::MatrixBase<Derived> &A, double *p, double pivtol = 1e-12) {
        int n = A.rows();

        for (int i = n - 1; i > 0; i--) {
            double piv = A(i, i - 1);

            if (std::abs(piv) < pivtol) {
                int piv_ind = 0;
                double piv_new = std::abs(piv);
                for (int j = 0; j < i - 1; j++) {
                    if (std::abs(A(i, j)) > piv_new) {
                        piv_new = std::abs(A(i, j));
                        piv_ind = j;
                    }
                }
                // Perform permutation
                A.row(i - 1).swap(A.row(piv_ind));
                A.col(i - 1).swap(A.col(piv_ind));
                piv = A(i, i - 1);
            }

            Eigen::VectorXd v = A.row(i);
            A.row(i - 1) = v.transpose()*A;

            Eigen::VectorXd vinv = (-1.0)*v;
            vinv(i - 1) = 1;
            vinv /= piv;
            vinv(i - 1) -= 1;
            Eigen::VectorXd Acol = A.col(i - 1);
            for (int j = 0; j <= i; j++)
                A.row(j) = A.row(j) + Acol(j)*vinv.transpose();


            A.row(i) = Eigen::VectorXd::Zero(n);
            A(i, i - 1) = 1;
        }
        p[n] = 1;
        for (int i = 0; i < n; i++)
            p[i] = -A(0, n - i - 1);
    }


/* Computes characteristic poly using Danilevsky's method with full pivoting */
    template<typename Derived>
    void charpoly_danilevsky_piv(Eigen::MatrixBase<Derived> &A, double *p) {
        int n = A.rows();

        for (int i = n - 1; i > 0; i--) {

            int piv_ind = i - 1;
            double piv = std::abs(A(i, i - 1));

            // Find largest pivot
            for (int j = 0; j < i - 1; j++) {
                if (std::abs(A(i, j)) > piv) {
                    piv = std::abs(A(i, j));
                    piv_ind = j;
                }
            }
            if (piv_ind != i - 1) {
                // Perform permutation
                A.row(i - 1).swap(A.row(piv_ind));
                A.col(i - 1).swap(A.col(piv_ind));
            }
            piv = A(i, i - 1);

            Eigen::VectorXd v = A.row(i);
            A.row(i - 1) = v.transpose()*A;

            Eigen::VectorXd vinv = (-1.0)*v;
            vinv(i - 1) = 1;
            vinv /= piv;
            vinv(i - 1) -= 1;
            Eigen::VectorXd Acol = A.col(i - 1);
            for (int j = 0; j <= i; j++)
                A.row(j) = A.row(j) + Acol(j)*vinv.transpose();


            A.row(i) = Eigen::VectorXd::Zero(n);
            A(i, i - 1) = 1;
        }
        p[n] = 1;
        for (int i = 0; i < n; i++)
            p[i] = -A(0, n - i - 1);
    }


/* Computes characteristic poly using Danilevsky's method.
   Also returns the transform T for the eigenvectors */
    template<typename Derived>
    void charpoly_danilevsky_T(Eigen::MatrixBase<Derived> &A, double *p, Eigen::MatrixBase<Derived> &T, double pivtol = 1e-12) {
        int n = A.rows();
        T.setIdentity();

        for (int i = n - 1; i > 0; i--) {
            double piv = A(i, i - 1);

            if (std::abs(piv) < pivtol) {
                int piv_ind = 0;
                double piv_new = std::abs(piv);
                for (int j = 0; j < i - 1; j++) {
                    if (std::abs(A(i, j)) > piv_new) {
                        piv_new = std::abs(A(i, j));
                        piv_ind = j;
                    }
                }
                // Perform permutation
                A.row(i - 1).swap(A.row(piv_ind));
                A.col(i - 1).swap(A.col(piv_ind));
                T.col(i - 1).swap(T.col(piv_ind));

                piv = A(i, i - 1);
            }

            Eigen::VectorXd v = A.row(i);
            A.row(i - 1) = v.transpose()*A;

            Eigen::VectorXd vinv = (-1.0)*v;
            vinv(i - 1) = 1;
            vinv /= piv;
            vinv(i - 1) -= 1;
            Eigen::VectorXd Acol = A.col(i - 1);
            for (int j = 0; j <= i; j++)
                A.row(j) = A.row(j) + Acol(j)*vinv.transpose();

            Eigen::VectorXd Tcol = T.col(i - 1);
            for (int j = 0; j < n - 1; j++)
                T.row(j) = T.row(j) + Tcol(j)*vinv.transpose();

            A.row(i) = Eigen::VectorXd::Zero(n);
            A(i, i - 1) = 1;
        }
        p[n] = 1;
        for (int i = 0; i < n; i++)
            p[i] = -A(0, n - i - 1);
    }

/* Computes characteristic poly using Danilevsky's method with full pivoting.
   Also returns the transform T for the eigenvectors */
    template<typename Derived>
    void charpoly_danilevsky_piv_T(Eigen::MatrixBase<Derived> &A, double *p, Eigen::MatrixBase<Derived> &T) {

        int n = A.rows();
        T.setIdentity();

        for (int i = n - 1; i > 0; i--) {

            int piv_ind = i - 1;
            double piv = std::abs(A(i, i - 1));

            // Find largest pivot
            for (int j = 0; j < i - 1; j++) {
                if (std::abs(A(i, j)) > piv) {
                    piv = std::abs(A(i, j));
                    piv_ind = j;
                }
            }
            if (piv_ind != i - 1) {
                // Perform permutation
                A.row(i - 1).swap(A.row(piv_ind));
                A.col(i - 1).swap(A.col(piv_ind));
                T.col(i - 1).swap(T.col(piv_ind));
            }
            piv = A(i, i - 1);

            Eigen::VectorXd v = A.row(i);
            A.row(i - 1) = v.transpose()*A;

            Eigen::VectorXd vinv = (-1.0)*v;
            vinv(i - 1) = 1;
            vinv /= piv;
            vinv(i - 1) -= 1;
            Eigen::VectorXd Acol = A.col(i - 1);
            for (int j = 0; j <= i; j++)
                A.row(j) = A.row(j) + Acol(j)*vinv.transpose();

            Eigen::VectorXd Tcol = T.col(i - 1);
            for (int j = 0; j < n - 1; j++)
                T.row(j) = T.row(j) + Tcol(j)*vinv.transpose();

            A.row(i) = Eigen::VectorXd::Zero(n);
            A(i, i - 1) = 1;
        }
        p[n] = 1;
        for (int i = 0; i < n; i++)
            p[i] = -A(0, n - i - 1);
    }


/* Computes characteristic poly using La Budde's method.
	https://arxiv.org/abs/1104.3769
 */
    template<typename Derived>
    void charpoly_la_budde(Eigen::MatrixBase<Derived> &A, double *p) {
        int n = A.rows();

        // Compute hessenberg form
        Eigen::HessenbergDecomposition<Eigen::MatrixXd> hess(A);
        Eigen::MatrixXd H = hess.matrixH();

        Eigen::VectorXd beta = H.diagonal(-1);
        Eigen::MatrixXd c(n, n);
        c.setZero();

        // Precompute the beta products needed
        Eigen::MatrixXd beta_prod(n - 1, n - 1);
        beta_prod.col(0) = beta;
        for (int i = 0; i < n - 1; i++) {
            for (int j = 1; j < n - i - 1; j++) {
                beta_prod(i, j) = beta_prod(i, j - 1) * beta(i + j);
            }
        }

        c(0, 0) = -H(0, 0);
        c(0, 1) = c(0, 0) - H(1, 1);
        c(1, 1) = H(0, 0)*H(1, 1) - H(0, 1)*beta(0);

        for (int i = 2; i < n; i++) {
            c(0, i) = c(0, i - 1) - H(i, i);
            for (int j = 1; j <= i - 1; j++) {
                c(j, i) = c(j, i - 1) - H(i, i)*c(j - 1, i - 1);
                for (int m = 1; m <= j - 1; m++) {
                    c(j, i) -= beta_prod(i - m, m - 1) * c(j - m - 1, i - m - 1) * H(i - m, i);
                }
                c(j, i) -= beta_prod(i - j, j - 1) * H(i - j, i);
            }
            c(i, i) = -H(i, i)*c(i - 1, i - 1);
            for (int m = 1; m <= i - 1; m++) {
                c(i, i) -= beta_prod(i - m, m - 1)*c(i - m - 1, i - m - 1)*H(i - m, i);
            }
            c(i, i) -= H(0, i)*beta_prod(0, i - 1);
        }

        p[n] = 1;
        for (int i = 0; i < n; i++)
            p[i] = c(n - i - 1, n - 1);
    }


// all real roots of a polynomial p(x) lie in (bound[0], bound[1])
// Kioustelidis' method
    static void getBounds (const double *p, double *bound, const int deg)
    {
        double M = fabs(p[0]);
        bound[0] = bound[1]=0.0;
        for (int i = 0; i < deg; ++i)
        {
            const double c = fabs(p[i]);
            if (c > M) M = c;
            if (p[i] < 0)
            {
                const double t = pow(-p[i], 1.0/(double)(deg - i));
                if (t > bound[1]) bound[1] = t;
                if (t > bound[0] && !(i%2)) bound[0] = t;
            }
            else if (i%2)
            {
                const double t = pow(p[i], 1.0/(double)(deg - i));
                if (t > bound[0]) bound[0] = t;
            }
        } // end for i
        M += 1.0; // Cauchy bound, since p[DEG] = 1.0;
        bound[0] *= -2.0;
        if (bound[0] < -M) bound[0] = -M;
        bound[1] *= 2.0;
        if (bound[1] > M) bound[1] = M;

    } // end getBounds()



// get data to compute the Sturm sequence for a polynomial p(x) at any point
//static void getQuotients (const double p[DEG], const double dp[DEG - 1], double q[DEG + 1][2])
    static void getQuotients (const double *p, const double *dp, double q[][2], const int deg)
    {
        double r_[deg], r[deg - 1];

        r_[deg - 1] = 1.0;
        q[deg][1] = p[deg - 1] - dp[deg - 2];
        r_[0] = dp[0];
        r[0] = dp[0]*q[deg][1] - p[0];
        for (int j = 1; j < deg - 1; ++j)
        {
            r_[j] = dp[j];
            r[j] = dp[j]*q[deg][1] + r_[j - 1] - p[j];
        }

        for (int i = deg - 1; i >= 2; --i)
        {
            const int i1 = i - 1;

#ifdef DEBUG_ERR
            {
			if (!r[i1])
				std::cout << "division by zero in getQuotients()\n";
		}
#endif

            const double ri1 = 1.0/r[i1];
            q[i][0] = r_[i]*ri1;
            q[i][1] = (r_[i1] - q[i][0]*r[i - 2])*ri1;
            const double t = r_[0];
            r_[0] = r[0];
            r[0] = r[0]*q[i][1] - t;
            for (int j = 1; j < i1; ++j)
            {
                const double t = r_[j];
                r_[j] = r[j];
                r[j] = r[j]*q[i][1] + q[i][0]*r_[j - 1] - t;
            }
            r_[i1] = r[i1];
        } // end for i

        q[1][0] = r_[1];
        q[1][1] = r_[0];
        q[0][1] = r[0];

    } // end getQuotients()



// evaluate polynomial p(x) at point x0
//static double evalPoly (const double p[DEG], const double &x0)
    static double evalPoly (const double *p, const double &x0, const int deg)
    {
        double s = x0 + p[deg - 1];
        for (int i = deg - 2; i >= 0; --i)
            s = s*x0 + p[i];
        return s;

    } // end evalPoly()



// compute the number of sign changes in a sequence seq[]
//static int nchanges (const double seq[DEG + 1])
    static int nchanges (const double *seq, const int deg)
    {
        int s = 0, s1, s2 = (seq[0] > 0.0)? 1 : ((seq[0] < 0.0)? -1 : 0);
        for (int i = 1; i < deg + 1 && seq[i]; ++i)
        {
            s1 = s2;
            s2 = (seq[i] > 0)? 1 : -1;
            if (!(s1 + s2)) ++s;
        }
        return s;

    } // end nchanges()



// evaluate Sturm sequence at point a
//static int evalSturmSeq (const double q[DEG + 1][2], const double &a)
    static int evalSturmSeq (const double q[][2], const double &a, const int deg)
    {
        double sa[deg + 1];
        // initialize sa
        sa[0] = q[0][1];
        sa[1] = q[1][0]*a + q[1][1];
        // compute sa recursively
        for (int i = 2; i < deg; ++i)
            sa[i] = (q[i][0]*a + q[i][1])*sa[i - 1] - sa[i - 2];
        sa[deg] = (a + q[deg][1])*sa[deg - 1] - sa[deg - 2]; // since q[DEG][0] = 1
        return nchanges(sa, deg);

    } // end evalSturmSeq()



// isolate all real roots of a given polynomial p(x)
//static int isolateRoots (const double p[DEG], const double dp[DEG - 1], double Isol[DEG][2])
    static int isolateRoots (const double *p, const double *dp, double Isol[][2], const int deg)
    {
        int nIsol = 0, nTree = 1, nIters = 1, min = 0, sTree[MAXTREE][2];
        double Tree[MAXTREE][2], q[deg + 1][2];

        // initialize the tree
        // all real roots of the polynomial p(x) lie in (Tree[0][0], Tree[0][1])
        getBounds(p, Tree[0], deg);

        getQuotients(p, dp, q, deg);
        sTree[0][0] = evalSturmSeq(q, Tree[0][0], deg);
        sTree[0][1] = evalSturmSeq(q, Tree[0][1], deg);

        while (nTree > min)
        {
            const double a = Tree[min][0], b = Tree[min][1];
            const int sa = sTree[min][0], sb = sTree[min][1];
            const int s = sa - sb; // counts the number of real roots in (a, b)
            ++min;

            if (s == 1) // an isolated root found
            {
                Isol[nIsol][0] = a;
                Isol[nIsol++][1] = b;
            }
            else if (s > 1) // proceed to make subdivision
            {
                const int nTree1 = nTree + 1;
                const double mid = 0.5*(a + b);
                // add intervals (a, mid] and (mid, b] to Isol
                Tree[nTree][1] = Tree[nTree1][0] = mid;
                sTree[nTree][1] = sTree[nTree1][0] = evalSturmSeq(q, mid, deg);
                Tree[nTree][0] = a;
                Tree[nTree1][1] = b;
                sTree[nTree][0] = sa;
                sTree[nTree1][1] = sb;
                nTree += 2;
                ++nIters;
                if (nIters > MAXIT)
                { // perhaps some roots are too close

#ifdef DEBUG_ERR
                    {
					std::cout << "isolateRoots() exceeds maximum iterations\n";
				}
#endif

                    Isol[nIsol][0] = a;
                    Isol[nIsol++][1] = mid;
                    Isol[nIsol][0] = mid;
                    Isol[nIsol++][1] = b;
                    return nIsol;
                }
            } // end else if (s > 1)
        } // end while (nTree > min)

        return nIsol;

    } // end isolateRoots()



// using Ridders' method, return the root of a polynomial p(x) known to lie between xl and x2
// this function is adopted from "Numerical recipes in C" by Press et al.
// the output is either 1, or 0 (no solution found)
//static bool polishRoots (const double p[DEG], const double &x1, const double &x2, double &ans)
    static bool polishRoots (const double *p, const double &x1, const double &x2, double &ans, const int deg)
    {
        double fl = evalPoly(p, x1, deg), fh = evalPoly(p, x2, deg);
        if (!fh)
        {
            ans = x2;
            return 1;
        }

        if ((fl > 0)? (fh < 0) : (fh > 0))
        {
            double xl = x1, xh = x2;
            //ans = xl;
            ans = 0.5*(x1 + x2);
            for (int j = 1; j <= MAXIT; ++j)
            {
                const double xm = 0.5*(xl + xh), fm = evalPoly(p, xm, deg), s = sqrt(fm*fm - fl*fh);
                if (!s) return 1;
                const double xnew = (fl < fh)? xm + (xl - xm)*fm/s : xm + (xm - xl)*fm/s;
                if (fabs(xnew - ans) <= ACC) return 1;
                ans = xnew;
                const double fnew = evalPoly(p, ans, deg);
                if (!fnew) return 1;
                if (fnew >= 0? (fm<0) : (fm > 0))
                {
                    xl = xm;
                    fl = fm;
                    xh = ans;
                    fh = fnew;
                }
                else if (fnew >= 0? (fl < 0) : (fl > 0))
                {
                    xh = ans;
                    fh = fnew;
                }
                else
                {
                    xl = ans;
                    fl = fnew;
                }
                if (fabs(xh - xl) <= ACC) return 1;
            } // end for j

#ifdef DEBUG_ERR
            {
			std::cout << "polishRoots() exceeds maximum iterations\n";
		}
#endif

            return 0;
        } // end if
        else
        {
#ifdef DEBUG_ERR
            {
			std::cout << "root must be bracketed in polishRoots()" << " " << x1 << " " << x2 << "\n";
		}
#endif

            return (fabs(fl) < fabs(fh))? x1 : x2;
            //return 0;
        }

    } // end polishRoots()



// find all real roots of the input square-free polynomial p(x) of degree DEG
//static int realRoots (const double p[DEG + 1], double roots[DEG])
    static int realRoots (const double *p, double *roots, const int deg)
    {
#ifdef DEBUG_ERR
        {
		if (!p[DEG])
			std::cout << "leading coefficient is zero in realRoots()\n";
	}
#endif

        // copy and normalize the input polynomial p(x) and its derivative dp(x)
        double p1[deg], dp1[deg - 1];
        const double pdeg = 1.0/p[deg], dpdeg = 1.0/(double)deg;
        p1[0] = p[0]*pdeg;
        for (int i = 1; i < deg; ++i)
        {
            p1[i] = p[i]*pdeg;
            dp1[i - 1] = (double)i*p1[i]*dpdeg;
        }

        double Isol[deg][2];

        // isolate all real roots of p(x)
        const int nIsol = isolateRoots(p1, dp1, Isol, deg);

        int nr = 0;
        for (int i = 0; i < nIsol; ++i)
        { // find an isolated real root of p(x)
            if (!polishRoots(p1, Isol[i][0], Isol[i][1], roots[nr], deg)) continue;
            ++nr;
        }

        return nr;

    } // end realRoots()



#undef MAXIT
#undef MAXTREE
#undef ACC
//#undef DEG

#undef DEBUG_ERR

    void relpose_kfEfk_fast_eigenvector_solver(double *eigv, int neig, Eigen::Matrix<double,113,113> &AM,
                                               Eigen::Matrix<std::complex<double>,2,113> &sols) {
        static const int ind[] = { 0,2,5,18,22,29,87,99 };
        // Truncated action matrix containing non-trivial rows
        Eigen::Matrix<double, 8, 113> AMs;
        double zi[16];

        for (int i = 0; i < 8; i++)	{
            AMs.row(i) = AM.row(ind[i]);
        }
        for (int i = 0; i < neig; i++) {
            zi[0] = eigv[i];
            for (int j = 1; j < 16; j++)
            {
                zi[j] = zi[j - 1] * eigv[i];
            }
            Eigen::Matrix<double, 8,8> AA;
            AA.col(0) = zi[14] * AMs.col(0) + zi[13] * AMs.col(1) + zi[12] * AMs.col(3) + zi[11] * AMs.col(6) + zi[10] * AMs.col(9) + zi[9] * AMs.col(12) + zi[8] * AMs.col(15) + zi[7] * AMs.col(19) + zi[6] * AMs.col(24) + zi[5] * AMs.col(30) + zi[4] * AMs.col(36) + zi[3] * AMs.col(42) + zi[2] * AMs.col(48) + zi[1] * AMs.col(54) + zi[0] * AMs.col(60) + AMs.col(66);
            AA.col(1) = zi[14] * AMs.col(2) + zi[13] * AMs.col(4) + zi[12] * AMs.col(7) + zi[11] * AMs.col(10) + zi[10] * AMs.col(13) + zi[9] * AMs.col(16) + zi[8] * AMs.col(20) + zi[7] * AMs.col(25) + zi[6] * AMs.col(31) + zi[5] * AMs.col(37) + zi[4] * AMs.col(43) + zi[3] * AMs.col(49) + zi[2] * AMs.col(55) + zi[1] * AMs.col(61) + zi[0] * AMs.col(67) + AMs.col(72);
            AA.col(2) = zi[14] * AMs.col(5) + zi[13] * AMs.col(8) + zi[12] * AMs.col(11) + zi[11] * AMs.col(14) + zi[10] * AMs.col(17) + zi[9] * AMs.col(21) + zi[8] * AMs.col(26) + zi[7] * AMs.col(32) + zi[6] * AMs.col(38) + zi[5] * AMs.col(44) + zi[4] * AMs.col(50) + zi[3] * AMs.col(56) + zi[2] * AMs.col(62) + zi[1] * AMs.col(68) + zi[0] * AMs.col(73) + AMs.col(77);
            AA.col(3) = zi[10] * AMs.col(22) + zi[9] * AMs.col(27) + zi[8] * AMs.col(33) + zi[7] * AMs.col(39) + zi[6] * AMs.col(45) + zi[5] * AMs.col(51) + zi[4] * AMs.col(57) + zi[3] * AMs.col(63) + zi[2] * AMs.col(69) + zi[1] * AMs.col(74) + zi[0] * AMs.col(78) + AMs.col(81);
            AA.col(4) = zi[12] * AMs.col(18) + zi[11] * AMs.col(23) + zi[10] * AMs.col(28) + zi[9] * AMs.col(34) + zi[8] * AMs.col(40) + zi[7] * AMs.col(46) + zi[6] * AMs.col(52) + zi[5] * AMs.col(58) + zi[4] * AMs.col(64) + zi[3] * AMs.col(70) + zi[2] * AMs.col(75) + zi[1] * AMs.col(79) + zi[0] * AMs.col(82) + AMs.col(84);
            AA.col(5) = zi[11] * AMs.col(29) + zi[10] * AMs.col(35) + zi[9] * AMs.col(41) + zi[8] * AMs.col(47) + zi[7] * AMs.col(53) + zi[6] * AMs.col(59) + zi[5] * AMs.col(65) + zi[4] * AMs.col(71) + zi[3] * AMs.col(76) + zi[2] * AMs.col(80) + zi[1] * AMs.col(83) + zi[0] * AMs.col(85) + AMs.col(86);
            AA.col(6) = zi[11] * AMs.col(87) + zi[10] * AMs.col(88) + zi[9] * AMs.col(89) + zi[8] * AMs.col(90) + zi[7] * AMs.col(91) + zi[6] * AMs.col(92) + zi[5] * AMs.col(93) + zi[4] * AMs.col(94) + zi[3] * AMs.col(95) + zi[2] * AMs.col(96) + zi[1] * AMs.col(97) + zi[0] * AMs.col(98) + AMs.col(110);
            AA.col(7) = zi[11] * AMs.col(99) + zi[10] * AMs.col(100) + zi[9] * AMs.col(101) + zi[8] * AMs.col(102) + zi[7] * AMs.col(103) + zi[6] * AMs.col(104) + zi[5] * AMs.col(105) + zi[4] * AMs.col(106) + zi[3] * AMs.col(107) + zi[2] * AMs.col(108) + zi[1] * AMs.col(109) + zi[0] * AMs.col(111) + AMs.col(112);
            AA(0,0) = AA(0,0) - zi[15];
            AA(1,1) = AA(1,1) - zi[15];
            AA(2,2) = AA(2,2) - zi[15];
            AA(4,3) = AA(4,3) - zi[11];
            AA(3,4) = AA(3,4) - zi[13];
            AA(5,5) = AA(5,5) - zi[12];
            AA(6,6) = AA(6,6) - zi[12];
            AA(7,7) = AA(7,7) - zi[12];


            Eigen::Matrix<double, 7, 1>  s = AA.leftCols(7).colPivHouseholderQr().solve(-AA.col(7));
//            Eigen::Matrix<double, 7, 1>  s = AA.leftCols(7).householderQr().solve(-AA.col(7));
            sols(1,i) = s(6);
            sols(0,i) = zi[0];
        }
    }

    Eigen::MatrixXcd solver_new_kfEfk(const double *d) {
        Eigen::VectorXd coeffs(167);

        coeffs[0] = d[0];
        coeffs[1] = d[11];
        coeffs[2] = d[23];
        coeffs[3] = d[36];
        coeffs[4] = d[1];
        coeffs[5] = d[12];
        coeffs[6] = d[24];
        coeffs[7] = d[37];
        coeffs[8] = d[2];
        coeffs[9] = d[13];
        coeffs[10] = d[25];
        coeffs[11] = d[38];
        coeffs[12] = d[3];
        coeffs[13] = d[14];
        coeffs[14] = d[26];
        coeffs[15] = d[39];
        coeffs[16] = d[4];
        coeffs[17] = d[15];
        coeffs[18] = d[27];
        coeffs[19] = d[40];
        coeffs[20] = d[5];
        coeffs[21] = d[16];
        coeffs[22] = d[28];
        coeffs[23] = d[41];
        coeffs[24] = d[6];
        coeffs[25] = d[17];
        coeffs[26] = d[29];
        coeffs[27] = d[42];
        coeffs[28] = d[7];
        coeffs[29] = d[18];
        coeffs[30] = d[30];
        coeffs[31] = d[43];
        coeffs[32] = d[8];
        coeffs[33] = d[19];
        coeffs[34] = d[31];
        coeffs[35] = d[44];
        coeffs[36] = d[9];
        coeffs[37] = d[20];
        coeffs[38] = d[32];
        coeffs[39] = d[45];
        coeffs[40] = d[10];
        coeffs[41] = d[21];
        coeffs[42] = d[33];
        coeffs[43] = d[46];
        coeffs[44] = d[22];
        coeffs[45] = d[34];
        coeffs[46] = d[47];
        coeffs[47] = d[35];
        coeffs[48] = d[48];
        coeffs[49] = d[49];
        coeffs[50] = d[50];
        coeffs[51] = d[67];
        coeffs[52] = d[85];
        coeffs[53] = d[104];
        coeffs[54] = d[124];
        coeffs[55] = d[145];
        coeffs[56] = d[51];
        coeffs[57] = d[68];
        coeffs[58] = d[86];
        coeffs[59] = d[105];
        coeffs[60] = d[125];
        coeffs[61] = d[146];
        coeffs[62] = d[52];
        coeffs[63] = d[69];
        coeffs[64] = d[87];
        coeffs[65] = d[106];
        coeffs[66] = d[126];
        coeffs[67] = d[147];
        coeffs[68] = d[53];
        coeffs[69] = d[70];
        coeffs[70] = d[88];
        coeffs[71] = d[107];
        coeffs[72] = d[127];
        coeffs[73] = d[148];
        coeffs[74] = d[54];
        coeffs[75] = d[71];
        coeffs[76] = d[89];
        coeffs[77] = d[108];
        coeffs[78] = d[128];
        coeffs[79] = d[149];
        coeffs[80] = d[55];
        coeffs[81] = d[72];
        coeffs[82] = d[90];
        coeffs[83] = d[109];
        coeffs[84] = d[129];
        coeffs[85] = d[150];
        coeffs[86] = d[56];
        coeffs[87] = d[73];
        coeffs[88] = d[91];
        coeffs[89] = d[110];
        coeffs[90] = d[130];
        coeffs[91] = d[151];
        coeffs[92] = d[57];
        coeffs[93] = d[74];
        coeffs[94] = d[92];
        coeffs[95] = d[111];
        coeffs[96] = d[131];
        coeffs[97] = d[152];
        coeffs[98] = d[58];
        coeffs[99] = d[75];
        coeffs[100] = d[93];
        coeffs[101] = d[112];
        coeffs[102] = d[132];
        coeffs[103] = d[153];
        coeffs[104] = d[59];
        coeffs[105] = d[76];
        coeffs[106] = d[94];
        coeffs[107] = d[113];
        coeffs[108] = d[133];
        coeffs[109] = d[154];
        coeffs[110] = d[60];
        coeffs[111] = d[77];
        coeffs[112] = d[95];
        coeffs[113] = d[114];
        coeffs[114] = d[134];
        coeffs[115] = d[155];
        coeffs[116] = d[61];
        coeffs[117] = d[78];
        coeffs[118] = d[96];
        coeffs[119] = d[115];
        coeffs[120] = d[135];
        coeffs[121] = d[156];
        coeffs[122] = d[62];
        coeffs[123] = d[79];
        coeffs[124] = d[97];
        coeffs[125] = d[116];
        coeffs[126] = d[136];
        coeffs[127] = d[157];
        coeffs[128] = d[63];
        coeffs[129] = d[80];
        coeffs[130] = d[98];
        coeffs[131] = d[117];
        coeffs[132] = d[137];
        coeffs[133] = d[158];
        coeffs[134] = d[64];
        coeffs[135] = d[81];
        coeffs[136] = d[99];
        coeffs[137] = d[118];
        coeffs[138] = d[138];
        coeffs[139] = d[159];
        coeffs[140] = d[65];
        coeffs[141] = d[82];
        coeffs[142] = d[100];
        coeffs[143] = d[119];
        coeffs[144] = d[139];
        coeffs[145] = d[160];
        coeffs[146] = d[66];
        coeffs[147] = d[83];
        coeffs[148] = d[101];
        coeffs[149] = d[120];
        coeffs[150] = d[140];
        coeffs[151] = d[161];
        coeffs[152] = d[84];
        coeffs[153] = d[102];
        coeffs[154] = d[121];
        coeffs[155] = d[141];
        coeffs[156] = d[162];
        coeffs[157] = d[103];
        coeffs[158] = d[122];
        coeffs[159] = d[142];
        coeffs[160] = d[163];
        coeffs[161] = d[123];
        coeffs[162] = d[143];
        coeffs[163] = d[164];
        coeffs[164] = d[144];
        coeffs[165] = d[165];
        coeffs[166] = d[166];

        // Setup elimination template
        static const int coeffs0_ind[] = { 0,1,50,51,1,0,2,50,51,52,2,1,0,3,51,52,53,3,2,1,0,52,53,54,3,2,1,53,54,55,3,2,54,55,3,55,5,4,6,56,57,0,1,2,50,58,51,6,5,4,7,57,58,1,0,2,3,51,59,52,7,6,5,4,58,59,2,1,0,3,52,60,53,7,6,5,59,60,3,2,1,53,61,54,7,6,60,61,3,2,54,55,7,61,3,55,10,9,8,11,63,64,5,4,6,7,57,0,1,2,3,51,65,58,11,10,9,8,64,65,6,5,4,7,58,1,0,2,3,52,66,59,11,10,9,65,66,7,6,5,59,2,1,3,53,67,60,11,10,66,67,7,6,60,3,2,54,61,11,67,7,61,3,55,14,13,12,15,69,70,9,8,10,11,63,4,5,6,7,1,2,3,0,57,71,64,15,14,13,12,70,71,10,9,8,11,64,5,4,6,7,2,0,3,1,58,72,65,15,14,13,71,72,11,10,9,65,6,5,7,3,1,2,59,73,66,15,14,72,73,11,10,66,7,6,2,3,60,67,15,73,11,67,7,3,61,18,17,16,19,75,76,13,12,14,15,69,8,9,10,11,5,6,7,4,1,2,0,3,63,77,70,19,18,17,16,76,77,14,13,12,15,70,9,8,10,11,6,4,7,5,2,3,1,64,78,71,0,19,18,17,77,78,15,14,13,71,10,9,11,7,5,6,3,2,65,79,72,1,19,18,78,79,15,14,72,11,10,6,7,3,66,73,2,19,79,15,73,11,7,67,3,22,21,20,23,81,82,17,16,18,19,75,12,13,14,15,9,10,11,8,5,6,4,7,69,2,1,3,0,83,76,23,22,21,20,82,83,18,17,16,19,76,13,12,14,15,10,8,11,9,6,7,5,70,3,2,1,0,84,77,4,23,22,21,83,84,19,18,17,77,14,13,15,11,9,10,7,6,71,3,2,1,85,78,5,23,22,84,85,19,18,78,15,14,10,11,7,72,3,2,79,6,23,85,19,79,15,11,73,3,7,26,25,24,27,87,88,21,20,22,23,81,16,17,18,19,13,14,15,12,9,10,8,11,75,6,5,7,4,1,3,0,2,89,82,27,26,25,89,90,23,22,21,83,18,17,19,15,13,14,11,10,77,7,6,5,3,1,2,91,84,9,27,26,90,91,23,22,84,19,18,14,15,11,78,7,6,2,3,85,10,27,91,23,85,19,15,79,7,3,11,31,30,29,95,96,27,26,25,89,22,21,23,19,17,18,15,14,83,11,10,9,7,5,6,2,3,1,97,90,13,31,30,96,97,27,26,90,23,22,18,19,15,84,11,10,6,7,3,2,91,14,31,97,27,91,23,19,85,11,7,3,15,35,34,102,103,31,30,96,27,26,22,23,19,90,15,14,10,11,7,6,2,3,97,18,35,103,31,97,27,23,91,15,11,7,3,19,39,109,35,103,31,27,97,19,15,11,7,3,23,30,29,28,31,93,94,25,24,26,27,87,20,21,22,23,17,18,19,16,13,14,12,15,81,10,9,11,8,5,7,4,0,1,6,95,88,2,0,50,4,5,56,0,1,57,50,9,8,10,62,63,4,5,6,56,0,1,2,50,64,57,27,26,25,24,88,89,22,21,20,23,82,17,16,18,19,14,12,15,13,10,11,9,76,7,6,5,4,2,0,1,3,90,83,8,35,34,33,101,102,31,30,29,95,26,25,27,23,21,22,19,18,89,15,14,13,11,9,10,6,7,5,1,2,103,96,17,3,39,38,108,109,35,34,102,31,30,26,27,23,96,19,18,14,15,11,10,6,7,2,103,22,3,43,115,39,109,35,31,103,23,19,15,11,7,27,3 };
        static const int coeffs1_ind[] = { 4,0,56,8,4,0,62,8,9,62,4,5,0,1,63,56,12,8,4,0,68,12,13,68,8,9,4,5,0,1,69,62,13,12,14,68,69,8,9,10,62,4,5,6,0,1,2,56,70,63,16,12,8,4,0,74,16,17,74,12,13,8,9,4,5,0,1,75,68,17,16,18,74,75,12,13,14,68,8,9,10,4,5,6,0,1,2,62,76,69,20,16,12,8,4,0,80,20,21,80,16,17,12,13,8,9,4,5,0,1,81,74,21,20,22,80,81,16,17,18,74,12,13,14,8,9,10,4,5,6,68,1,0,2,82,75,24,20,16,12,8,4,0,86,24,25,86,20,21,16,17,12,13,8,9,4,5,1,0,87,80,25,24,26,86,87,20,21,22,80,16,17,18,12,13,14,8,9,10,74,5,4,6,0,2,1,88,81,28,24,20,16,12,8,4,92,28,29,92,24,25,20,21,16,17,12,13,8,9,5,4,93,86,0,29,28,30,92,93,24,25,26,86,20,21,22,16,17,18,12,13,14,80,9,8,10,4,6,0,5,94,87,1,31,30,29,28,94,95,26,25,24,27,88,21,20,22,23,18,16,19,17,14,15,13,82,11,10,9,8,6,4,5,1,2,0,7,96,89,12,3,32,28,24,20,16,12,8,98,32,33,98,28,29,24,25,20,21,16,17,12,13,9,8,99,92,4,33,32,34,98,99,28,29,30,92,24,25,26,20,21,22,16,17,18,86,13,12,14,8,10,4,9,100,93,5,0,34,33,32,35,99,100,29,28,30,31,93,24,25,26,27,21,22,23,20,17,18,16,19,87,14,13,15,12,9,11,8,4,5,0,10,101,94,6,1,35,34,33,32,100,101,30,29,28,31,94,25,24,26,27,22,20,23,21,18,19,17,88,15,14,13,12,10,8,9,5,6,4,0,1,11,102,95,16,7,2,36,32,28,24,20,16,12,104,36,37,104,32,33,28,29,24,25,20,21,16,17,13,12,105,98,8,37,36,38,104,105,32,33,34,98,28,29,30,24,25,26,20,21,22,92,17,16,18,12,14,8,13,106,99,9,4,38,37,36,39,105,106,33,32,34,35,99,28,29,30,31,25,26,27,24,21,22,20,23,93,18,17,19,16,13,15,12,8,9,4,14,107,100,10,5,0,39,38,37,36,106,107,34,33,32,35,100,29,28,30,31,26,24,27,25,22,23,21,94,19,18,17,16,14,12,13,9,10,8,4,5,0,15,108,101,20,11,6,1,39,38,37,107,108,35,34,33,101,30,29,31,27,25,26,23,22,95,19,18,17,15,13,14,10,11,9,5,6,1,109,102,21,7,2,40,36,32,28,24,20,16,110,40,41,110,36,37,32,33,28,29,24,25,20,21,17,16,111,104,12,41,40,42,110,111,36,37,38,104,32,33,34,28,29,30,24,25,26,98,21,20,22,16,18,12,17,112,105,13,8,42,41,40,43,111,112,37,36,38,39,105,32,33,34,35,29,30,31,28,25,26,24,27,99,22,21,23,20,17,19,16,12,13,8,18,113,106,14,9,4,43,42,41,40,112,113,38,37,36,39,106,33,32,34,35,30,28,31,29,26,27,25,100,23,22,21,20,18,16,17,13,14,12,8,9,4,19,114,107,24,15,10,5,0,43,42,41,113,114,39,38,37,107,34,33,35,31,29,30,27,26,101,23,22,21,19,17,18,14,15,13,9,10,5,115,108,25,11,6,1,40,36,32,28,24,20,116,44,116,40,41,36,37,32,33,28,29,24,25,21,20,117,110,16,44,45,116,117,40,41,42,110,36,37,38,32,33,34,28,29,30,104,25,24,26,20,22,16,21,118,111,17,12,45,44,46,117,118,41,40,42,43,111,36,37,38,39,33,34,35,32,29,30,28,31,105,26,25,27,24,21,23,20,16,17,12,22,119,112,18,13,8,46,45,44,118,119,42,41,40,43,112,37,36,38,39,34,32,35,33,30,31,29,106,27,26,25,24,22,20,21,17,18,16,12,13,8,23,120,113,28,19,14,9,4,46,45,44,119,120,43,42,41,113,38,37,39,35,33,34,31,30,107,27,26,25,23,21,22,18,19,17,13,14,9,121,114,29,15,10,5,40,36,32,28,24,122,122,44,40,41,36,37,32,33,28,29,25,24,123,116,20,47,122,123,44,45,116,40,41,42,36,37,38,32,33,34,110,29,28,30,24,26,20,25,124,117,21,16,47,48,123,124,44,45,46,117,40,41,42,43,37,38,39,36,33,34,32,35,111,30,29,31,28,25,27,24,20,21,16,26,125,118,22,17,12,48,47,124,125,45,44,46,118,41,40,42,43,38,36,39,37,34,35,33,112,31,30,29,28,26,24,25,21,22,20,16,17,12,27,126,119,32,23,18,13,8,48,47,125,126,46,45,44,119,42,41,43,39,37,38,35,34,113,31,30,29,27,25,26,22,23,21,17,18,13,127,120,33,19,14,9,40,36,32,28,128,128,44,40,41,36,37,32,33,29,28,129,122,24,128,129,47,122,44,45,40,41,42,36,37,38,116,33,32,34,28,30,24,29,130,123,25,20,49,129,130,47,48,123,44,45,46,41,42,43,40,37,38,36,39,117,34,33,35,32,29,31,28,24,25,20,30,131,124,26,21,16,49,130,131,47,48,124,44,45,46,42,40,43,41,38,39,37,118,35,34,33,32,30,28,29,25,26,24,20,21,16,31,132,125,36,27,22,17,12,49,131,132,48,47,125,45,44,46,43,41,42,39,38,119,35,34,33,31,29,30,26,27,25,21,22,17,133,126,37,23,18,13,40,36,32,134,134,44,40,41,36,37,33,32,135,128,28,134,135,128,47,44,45,40,41,42,122,37,36,38,32,34,28,33,136,129,29,24,135,136,49,129,47,48,44,45,46,41,42,40,43,123,38,37,39,36,33,35,32,28,29,24,34,137,130,30,25,20,136,137,49,130,47,48,45,46,44,42,43,41,124,39,38,37,36,34,32,33,29,30,28,24,25,20,35,138,131,40,31,26,21,16,137,138,49,131,47,48,46,44,45,43,42,125,39,38,37,35,33,34,30,31,29,25,26,21,139,132,41,27,22,17,40,36,140,140,44,40,41,37,36,141,134,32,140,141,134,47,44,45,128,41,40,42,36,38,32,37,142,135,33,28,141,142,135,49,47,48,44,45,46,129,42,41,43,40,37,39,36,32,33,28,38,143,136,34,29,24,142,143,136,49,47,48,45,46,44,130,43,42,41,40,38,36,37,33,34,32,28,29,24,39,144,137,35,30,25,20,143,144,137,49,48,47,46,45,131,43,42,41,39,37,38,34,35,33,29,30,25,145,138,44,31,26,21,40,146,146,44,41,40,147,140,36,146,147,140,47,134,44,45,40,42,36,41,148,141,37,32,147,148,141,49,47,48,135,45,44,46,41,43,40,36,37,32,42,149,142,38,33,28,148,149,142,49,47,48,136,46,45,44,42,40,41,37,38,36,32,33,28,43,150,143,39,34,29,24,149,150,143,49,48,47,137,46,45,44,43,41,42,38,39,37,33,34,29,151,144,35,30,25,44,152,146,40,152,146,140,47,45,40,44,153,147,41,36,152,153,147,49,141,47,48,44,46,40,41,36,45,154,148,42,37,32,153,154,148,49,142,48,47,45,44,41,42,40,36,37,32,46,155,149,43,38,33,28,154,155,149,49,143,48,47,46,44,45,42,43,41,37,38,33,156,150,39,34,29,146,47,157,152,44,40,157,152,147,49,48,44,40,47,158,153,45,41,36,157,158,153,148,49,47,44,45,40,41,36,48,159,154,46,42,37,32,158,159,154,149,49,48,47,45,46,44,41,42,37,160,155,43,38,33,152,49,161,157,47,44,40,161,157,153,47,44,40,49,162,158,48,45,41,36,161,162,158,154,49,47,48,44,45,41,163,159,46,42,37,157,164,161,49,47,44,40,164,161,158,49,47,44,165,162,48,45,41,161,166,164,49,47,44,43,42,114,115,39,38,108,35,34,30,31,27,102,23,22,18,19,15,14,10,11,6,109,26,7,2,46,45,120,121,43,42,114,39,38,34,35,31,108,27,26,22,23,19,18,14,15,10,115,30,11,6,48,47,126,127,46,45,120,43,42,38,39,35,114,31,30,26,27,23,22,18,19,14,121,34,15,10,49,132,133,48,47,126,46,45,42,43,39,120,35,34,30,31,27,26,22,23,18,127,38,19,14,138,139,49,132,48,47,45,46,43,126,39,38,34,35,31,30,26,27,22,133,42,23,18,144,145,138,49,47,48,46,132,43,42,38,39,35,34,30,31,26,139,45,27,22,150,151,144,49,48,138,46,45,42,43,39,38,34,35,30,145,47,31,26,155,156,150,49,144,48,47,45,46,43,42,38,39,34,151,35,30,159,160,155,150,49,47,48,46,45,42,43,38,156,39,34,162,163,159,155,49,48,47,45,46,42,160,43,38,164,165,162,159,49,47,48,45,163,46,42,166,164,162,49,47,165,48,45,46,121,43,115,39,35,109,27,23,19,15,11,31,7,48,127,46,121,43,39,115,31,27,23,19,15,35,11,49,133,48,127,46,43,121,35,31,27,23,19,39,15,139,49,133,48,46,127,39,35,31,27,23,43,19,145,139,49,48,133,43,39,35,31,27,46,23,151,145,49,139,46,43,39,35,31,48,27,156,151,145,48,46,43,39,35,49,31,160,156,151,49,48,46,43,39,35,163,160,156,49,48,46,43,39,165,163,160,49,48,46,43,166,165,163,49,48,46,164,166,49,47,166,165,49,48,166,49 };

        static const int C0_ind[] = { 0,4,6,44,51,52,55,56,57,95,102,103,104,106,107,108,146,153,154,155,156,158,159,197,205,206,207,209,210,248,257,258,260,261,309,311,357,358,361,362,363,364,367,368,369,401,402,408,409,410,412,413,414,415,416,418,419,420,452,453,459,460,461,462,464,465,466,467,468,469,471,503,504,511,512,513,515,516,517,518,519,522,554,555,563,564,566,567,569,570,573,606,615,617,621,624,663,664,665,667,668,669,670,671,673,674,675,676,678,679,680,690,707,708,714,715,716,717,719,720,721,722,723,724,726,727,728,729,730,741,758,759,766,767,768,770,771,772,773,774,777,778,779,780,792,809,810,818,819,821,822,824,825,828,829,830,843,861,870,872,876,879,881,894,918,919,920,922,923,924,925,926,928,929,930,931,933,934,935,936,938,939,940,945,962,963,969,970,971,972,974,975,976,977,978,979,981,982,983,984,985,987,988,989,991,996,1013,1014,1021,1022,1023,1025,1026,1027,1028,1029,1032,1033,1034,1035,1038,1039,1042,1047,1064,1065,1073,1074,1076,1077,1079,1080,1083,1084,1085,1090,1093,1098,1116,1125,1127,1131,1134,1136,1141,1149,1173,1174,1175,1177,1178,1179,1180,1181,1183,1184,1185,1186,1188,1189,1190,1191,1193,1194,1195,1196,1197,1198,1199,1200,1217,1218,1224,1225,1226,1227,1229,1230,1231,1232,1233,1234,1236,1237,1238,1239,1240,1242,1243,1244,1246,1247,1248,1249,1251,1268,1269,1270,1276,1277,1278,1280,1281,1282,1283,1284,1287,1288,1289,1290,1293,1294,1297,1298,1300,1302,1319,1320,1321,1328,1329,1331,1332,1334,1335,1338,1339,1340,1345,1348,1351,1353,1371,1372,1380,1382,1386,1389,1391,1396,1404,1423,1428,1429,1430,1432,1433,1434,1435,1436,1438,1439,1440,1441,1443,1444,1445,1446,1448,1449,1450,1451,1452,1453,1454,1455,1456,1457,1458,1459,1472,1473,1479,1480,1481,1482,1484,1485,1486,1487,1488,1489,1491,1492,1493,1494,1495,1497,1498,1499,1501,1502,1503,1504,1506,1507,1508,1510,1511,1523,1524,1525,1531,1532,1533,1535,1536,1537,1538,1539,1542,1543,1544,1545,1548,1549,1552,1553,1555,1557,1559,1561,1562,1574,1575,1576,1583,1584,1586,1587,1589,1590,1593,1594,1595,1600,1603,1606,1608,1612,1613,1626,1627,1635,1637,1641,1644,1646,1651,1659,1664,1678,1683,1684,1685,1687,1688,1689,1690,1691,1693,1694,1695,1696,1698,1699,1700,1701,1703,1704,1705,1706,1707,1708,1709,1710,1711,1712,1713,1714,1716,1717,1719,1726,1727,1728,1735,1736,1737,1739,1740,1741,1742,1743,1746,1747,1748,1749,1752,1753,1756,1757,1759,1761,1763,1765,1766,1767,1769,1770,1778,1779,1780,1787,1788,1790,1791,1793,1794,1797,1798,1799,1804,1807,1810,1812,1816,1817,1820,1821,1830,1831,1839,1841,1845,1848,1850,1855,1863,1868,1871,1882,1888,1889,1890,1892,1893,1894,1895,1896,1899,1900,1901,1902,1905,1906,1909,1910,1912,1914,1916,1918,1919,1920,1922,1923,1924,1925,1926,1931,1932,1933,1940,1941,1943,1944,1946,1947,1950,1951,1952,1957,1960,1963,1965,1969,1970,1973,1974,1975,1977,1983,1984,1992,1994,1998,2001,2003,2008,2016,2021,2024,2028,2035,2042,2043,2045,2046,2048,2049,2052,2053,2054,2059,2062,2065,2067,2071,2072,2075,2076,2077,2079,2080,2081,2085,2086,2094,2096,2100,2103,2105,2110,2118,2123,2126,2130,2131,2137,2145,2147,2151,2154,2156,2161,2169,2174,2177,2181,2182,2184,2188,2193,2194,2195,2197,2198,2199,2200,2201,2203,2204,2205,2206,2208,2209,2210,2211,2213,2214,2215,2216,2217,2218,2219,2220,2221,2222,2223,2224,2226,2227,2229,2230,2231,2236,2237,2238,2240,2248,2288,2295,2299,2301,2305,2306,2339,2340,2346,2347,2350,2351,2352,2353,2356,2357,2358,2361,2362,2363,2373,2390,2391,2397,2398,2399,2400,2402,2403,2404,2405,2406,2407,2409,2410,2411,2412,2413,2415,2416,2417,2419,2420,2421,2422,2424,2425,2426,2428,2429,2430,2432,2433,2440,2441,2442,2443,2449,2450,2451,2453,2454,2455,2456,2457,2460,2461,2462,2463,2466,2467,2470,2471,2473,2475,2477,2479,2480,2481,2483,2484,2485,2486,2487,2488,2489,2492,2493,2494,2496,2501,2502,2504,2505,2507,2508,2511,2512,2513,2518,2521,2524,2526,2530,2531,2534,2535,2536,2538,2539,2540,2541,2544,2545,2548,2553,2555,2559,2562,2564,2569,2577,2582,2585,2589,2590,2592,2596,2600 } ;
        static const int C1_ind[] = { 4,11,44,55,62,68,95,102,106,108,112,113,118,119,146,147,157,164,170,174,197,204,208,210,214,215,220,221,224,225,248,249,255,256,259,260,261,262,265,266,267,270,271,272,273,275,276,282,299,300,310,317,323,327,332,350,357,361,363,367,368,373,374,377,378,381,383,401,402,408,409,412,413,414,415,418,419,420,423,424,425,426,428,429,431,432,434,435,452,453,463,470,476,480,485,489,503,510,514,516,520,521,526,527,530,531,534,536,538,540,554,555,561,562,565,566,567,568,571,572,573,576,577,578,579,581,582,584,585,587,588,589,590,591,605,606,616,623,629,633,638,642,646,656,663,667,669,673,674,679,680,683,684,687,689,691,693,697,706,707,708,714,715,718,719,720,721,724,725,726,729,730,731,732,734,735,737,738,740,741,742,743,744,747,748,757,758,759,769,776,782,786,791,795,799,809,816,820,822,826,827,832,833,836,837,840,842,844,846,850,859,860,861,863,867,868,871,872,873,874,877,878,879,882,883,884,885,887,888,890,891,893,894,895,896,897,900,901,905,910,911,912,914,918,919,920,921,923,924,925,926,927,928,930,931,932,933,934,936,937,938,940,941,942,943,945,946,947,949,950,951,953,954,955,956,957,961,962,963,964,965,973,980,986,990,995,999,1003,1013,1020,1024,1026,1030,1031,1036,1037,1040,1041,1044,1046,1048,1050,1054,1063,1064,1065,1067,1071,1072,1075,1076,1077,1078,1081,1082,1083,1086,1087,1088,1089,1091,1092,1094,1095,1097,1098,1099,1100,1101,1104,1105,1109,1114,1115,1116,1118,1119,1122,1123,1124,1126,1127,1128,1129,1130,1132,1133,1134,1135,1137,1138,1139,1140,1142,1143,1144,1145,1146,1147,1148,1149,1150,1151,1152,1153,1155,1156,1158,1159,1160,1163,1165,1166,1167,1169,1170,1173,1174,1175,1176,1178,1179,1180,1181,1182,1183,1185,1186,1187,1188,1189,1191,1192,1193,1195,1196,1197,1198,1200,1201,1202,1204,1205,1206,1208,1209,1210,1211,1212,1213,1214,1216,1217,1218,1219,1220,1221,1228,1235,1241,1245,1250,1254,1258,1268,1275,1279,1281,1285,1286,1291,1292,1295,1296,1299,1301,1303,1305,1309,1318,1319,1320,1322,1326,1327,1330,1331,1332,1333,1336,1337,1338,1341,1342,1343,1344,1346,1347,1349,1350,1352,1353,1354,1355,1356,1359,1360,1364,1369,1370,1371,1373,1374,1377,1378,1379,1381,1382,1383,1384,1385,1387,1388,1389,1390,1392,1393,1394,1395,1397,1398,1399,1400,1401,1402,1403,1404,1405,1406,1407,1408,1410,1411,1413,1414,1415,1418,1420,1421,1422,1424,1425,1426,1428,1429,1430,1431,1433,1434,1435,1436,1437,1438,1440,1441,1442,1443,1444,1446,1447,1448,1450,1451,1452,1453,1455,1456,1457,1459,1460,1461,1463,1464,1465,1466,1467,1468,1469,1470,1471,1472,1473,1474,1475,1476,1477,1480,1481,1482,1484,1485,1486,1487,1488,1491,1492,1493,1494,1497,1498,1501,1502,1504,1506,1508,1510,1511,1512,1514,1515,1516,1517,1518,1519,1520,1521,1523,1524,1525,1527,1528,1534,1541,1547,1551,1556,1560,1564,1574,1581,1585,1587,1591,1592,1597,1598,1601,1602,1605,1607,1609,1611,1615,1624,1625,1626,1628,1632,1633,1636,1637,1638,1639,1642,1643,1644,1647,1648,1649,1650,1652,1653,1655,1656,1658,1659,1660,1661,1662,1665,1666,1670,1675,1676,1677,1679,1680,1683,1684,1685,1687,1688,1689,1690,1691,1693,1694,1695,1696,1698,1699,1700,1701,1703,1704,1705,1706,1707,1708,1709,1710,1711,1712,1713,1714,1716,1717,1719,1720,1721,1724,1726,1727,1728,1730,1731,1732,1734,1735,1736,1737,1739,1740,1741,1742,1743,1744,1746,1747,1748,1749,1750,1752,1753,1754,1756,1757,1758,1759,1761,1762,1763,1765,1766,1767,1769,1770,1771,1772,1773,1774,1775,1776,1777,1778,1779,1780,1781,1782,1783,1784,1786,1787,1788,1790,1791,1792,1793,1794,1797,1798,1799,1800,1803,1804,1807,1808,1810,1812,1814,1816,1817,1818,1820,1821,1822,1823,1824,1825,1826,1827,1829,1830,1831,1833,1834,1835,1847,1853,1857,1862,1866,1870,1880,1891,1893,1897,1898,1903,1904,1907,1908,1911,1913,1915,1917,1921,1930,1931,1932,1934,1938,1942,1943,1944,1945,1948,1949,1950,1953,1954,1955,1956,1958,1959,1961,1962,1964,1965,1966,1967,1968,1971,1972,1976,1981,1982,1983,1985,1986,1989,1990,1993,1994,1995,1996,1997,1999,2000,2001,2002,2004,2005,2006,2007,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2022,2023,2025,2026,2027,2030,2032,2033,2034,2036,2037,2038,2040,2041,2042,2045,2046,2047,2048,2049,2050,2052,2053,2054,2055,2056,2058,2059,2060,2062,2063,2064,2065,2067,2068,2069,2071,2072,2073,2075,2076,2077,2078,2079,2080,2081,2082,2083,2084,2085,2086,2087,2088,2089,2090,2092,2093,2094,2096,2097,2098,2099,2100,2103,2104,2105,2106,2109,2110,2113,2114,2116,2118,2120,2122,2123,2124,2126,2127,2128,2129,2130,2131,2132,2133,2135,2136,2137,2139,2140,2141,2159,2163,2168,2172,2176,2186,2199,2204,2209,2210,2213,2214,2217,2219,2221,2223,2227,2236,2237,2238,2240,2248,2249,2250,2254,2255,2256,2259,2260,2261,2262,2264,2265,2267,2268,2270,2271,2272,2273,2274,2277,2278,2282,2287,2288,2289,2291,2292,2295,2299,2300,2301,2302,2305,2306,2307,2308,2310,2311,2312,2313,2315,2316,2317,2318,2319,2320,2321,2322,2323,2324,2325,2326,2328,2329,2331,2332,2333,2336,2338,2339,2340,2342,2343,2344,2346,2347,2351,2352,2353,2354,2356,2358,2359,2360,2361,2362,2364,2365,2366,2368,2369,2370,2371,2373,2374,2375,2377,2378,2379,2381,2382,2383,2384,2385,2386,2387,2388,2389,2390,2391,2392,2393,2394,2395,2396,2398,2399,2402,2403,2404,2405,2406,2409,2410,2411,2412,2415,2416,2419,2420,2422,2424,2426,2428,2429,2430,2432,2433,2434,2435,2436,2437,2438,2439,2441,2442,2443,2445,2446,2447,2469,2474,2478,2482,2492,2505,2516,2519,2520,2523,2525,2527,2529,2533,2542,2543,2544,2546,2555,2556,2561,2562,2566,2567,2568,2570,2571,2573,2574,2576,2577,2578,2579,2580,2583,2584,2588,2593,2594,2595,2597,2598,2605,2606,2607,2611,2612,2613,2616,2617,2618,2619,2621,2622,2623,2624,2625,2626,2627,2628,2629,2630,2631,2632,2634,2635,2637,2638,2639,2642,2644,2645,2646,2648,2649,2650,2652,2657,2658,2659,2662,2664,2665,2667,2668,2670,2671,2672,2674,2675,2676,2677,2679,2680,2681,2683,2684,2685,2687,2688,2689,2690,2691,2692,2693,2694,2695,2696,2697,2698,2699,2700,2701,2702,2704,2708,2709,2710,2711,2715,2716,2717,2718,2721,2722,2725,2726,2728,2730,2732,2734,2735,2736,2738,2739,2740,2741,2742,2743,2744,2745,2747,2748,2749,2751,2752,2753,2780,2784,2788,2798,2811,2826,2829,2831,2833,2835,2839,2848,2849,2850,2852,2861,2862,2868,2873,2876,2877,2879,2880,2882,2883,2884,2885,2886,2889,2890,2894,2899,2900,2901,2903,2904,2912,2913,2918,2919,2923,2924,2925,2927,2928,2930,2931,2932,2933,2934,2935,2936,2937,2938,2940,2941,2943,2944,2945,2948,2950,2951,2952,2954,2955,2956,2963,2964,2968,2970,2973,2974,2976,2978,2980,2981,2982,2983,2985,2986,2987,2989,2990,2991,2993,2994,2995,2996,2997,2998,2999,3000,3001,3002,3003,3004,3005,3006,3007,3008,3014,3015,3016,3021,3022,3024,3027,3028,3031,3032,3034,3036,3038,3040,3041,3042,3044,3045,3046,3047,3048,3049,3050,3051,3053,3054,3055,3057,3058,3059,3090,3094,3104,3117,3137,3139,3141,3145,3154,3155,3156,3158,3167,3168,3174,3183,3186,3188,3189,3190,3191,3192,3195,3196,3200,3205,3206,3207,3209,3210,3218,3219,3225,3230,3233,3234,3236,3237,3239,3240,3241,3242,3243,3244,3246,3247,3249,3250,3251,3254,3256,3257,3258,3260,3261,3262,3269,3270,3276,3280,3282,3284,3287,3288,3289,3291,3292,3293,3295,3296,3297,3299,3300,3301,3302,3303,3304,3305,3306,3307,3308,3309,3311,3312,3313,3314,3320,3321,3327,3330,3333,3337,3338,3340,3342,3344,3346,3347,3348,3350,3351,3352,3353,3354,3355,3356,3357,3359,3360,3361,3363,3364,3365,3400,3410,3423,3447,3451,3460,3461,3462,3464,3473,3474,3480,3494,3495,3496,3498,3501,3502,3506,3511,3512,3513,3515,3516,3524,3525,3531,3540,3543,3545,3546,3547,3548,3549,3552,3553,3555,3556,3557,3560,3562,3563,3564,3566,3567,3568,3575,3576,3582,3590,3593,3594,3597,3598,3599,3601,3603,3605,3606,3607,3608,3609,3610,3611,3612,3613,3614,3615,3617,3618,3619,3620,3626,3627,3633,3639,3644,3646,3648,3650,3652,3653,3654,3656,3657,3658,3659,3660,3661,3662,3663,3665,3666,3669,3670,3671,3706,3716,3717,3719,3729,3735,3750,3753,3757,3761,3766,3767,3768,3770,3771,3779,3780,3786,3800,3801,3802,3804,3807,3808,3811,3812,3815,3817,3818,3819,3821,3822,3823,3830,3831,3837,3849,3852,3853,3854,3858,3861,3862,3863,3864,3865,3866,3867,3868,3869,3870,3872,3873,3874,3875,3881,3882,3888,3899,3903,3905,3907,3909,3911,3912,3913,3914,3915,3916,3917,3918,3920,3921,3924,3925,3926,3954,3961,3971,3972,3974,3975,3984,3990,4005,4008,4012,4016,4019,4021,4022,4023,4025,4026,4027,4034,4035,4041,4056,4057,4062,4066,4067,4069,4070,4071,4072,4073,4074,4076,4077,4078,4079,4085,4086,4092,4107,4109,4113,4116,4117,4118,4119,4120,4121,4122,4124,4125,4128,4129,4130,4158,4165,4175,4176,4178,4179,4180,4188,4194,4209,4220,4223,4224,4225,4226,4227,4229,4230,4231,4232,4238,4239,4245,4260,4266,4270,4271,4273,4274,4275,4277,4278,4281,4282,4283,4311,4328,4329,4331,4332,4333,4334,4341,4347,4362,4373,4376,4377,4379,4380,4383,4384,4385,4413,4430,4431,4434,4435,4436,4439,4440,4442,4443,4445,4446,4449,4450,4451,4456,4459,4462,4464,4468,4469,4472,4473,4474,4476,4477,4478,4479,4482,4483,4486,4487,4490,4491,4493,4494,4496,4497,4500,4501,4502,4507,4510,4513,4515,4519,4520,4523,4524,4525,4527,4528,4529,4530,4533,4534,4537,4538,4541,4542,4544,4545,4547,4548,4551,4552,4553,4558,4561,4564,4566,4570,4571,4574,4575,4576,4578,4579,4580,4581,4584,4585,4588,4589,4592,4595,4596,4598,4599,4602,4603,4604,4609,4612,4615,4617,4621,4622,4625,4626,4627,4629,4630,4631,4632,4635,4636,4639,4640,4646,4647,4649,4653,4654,4655,4660,4663,4666,4668,4672,4673,4676,4677,4678,4680,4681,4682,4683,4686,4687,4690,4691,4697,4698,4704,4705,4711,4714,4717,4719,4723,4724,4727,4728,4729,4731,4732,4733,4734,4737,4738,4741,4742,4748,4749,4755,4765,4768,4770,4774,4775,4778,4779,4780,4782,4783,4784,4785,4788,4789,4792,4793,4799,4800,4806,4819,4821,4825,4826,4829,4830,4831,4833,4834,4835,4836,4839,4843,4844,4850,4851,4857,4872,4876,4880,4881,4882,4884,4885,4886,4887,4890,4894,4895,4901,4902,4908,4923,4932,4933,4935,4936,4937,4938,4941,4945,4946,4952,4953,4959,4974,4984,4987,4988,4989,4992,4996,4997,5004,5010,5025,5039,5040,5043,5047,5048,5052,5054,5058,5061,5063,5068,5076,5081,5084,5088,5089,5091,5095,5099,5103,5105,5109,5112,5114,5119,5127,5132,5135,5139,5140,5142,5146,5150,5154,5156,5160,5163,5165,5170,5178,5183,5186,5190,5191,5193,5197,5201,5207,5211,5214,5216,5221,5229,5234,5237,5241,5242,5244,5248,5252,5258,5265,5267,5272,5280,5285,5288,5292,5293,5295,5299,5303,5309,5316,5323,5331,5336,5339,5343,5344,5346,5350,5354,5360,5367,5382,5387,5390,5394,5395,5397,5401,5405,5411,5418,5433,5438,5441,5445,5446,5448,5456,5462,5469,5484,5492,5496,5497,5499,5507,5513,5520,5535,5547,5548,5550,5558,5564,5571,5586,5599,5601,5609,5637,5655,5659,5660,5673,5688,5703,5711,5739,5762 };

        Eigen::MatrixXd C0 = Eigen::MatrixXd::Zero(51,51);
        Eigen::MatrixXd C1 = Eigen::MatrixXd::Zero(51,113);
        for (int i = 0; i < 819; i++) { C0(C0_ind[i]) = coeffs(coeffs0_ind[i]); }
        for (int i = 0; i < 2133; i++) { C1(C1_ind[i]) = coeffs(coeffs1_ind[i]); }

        Eigen::MatrixXd C12 = C0.partialPivLu().solve(C1);
//        Eigen::MatrixXd C12 = C0.fullPivLu().solve(C1);

        // Setup action matrix
        Eigen::Matrix<double,121, 113> RR;
        RR << -C12.bottomRows(8), Eigen::Matrix<double,113,113>::Identity(113, 113);

        static const int AM_ind[] = { 1,8,2,9,10,3,11,12,13,14,15,16,17,18,19,20,21,22,4,23,24,25,0,26,27,28,29,30,31,5,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,75,76,77,78,79,81,82,83,84,86,87,88,90,91,93,6,95,96,97,98,99,100,101,102,103,104,105,7,107,108,109,110,111,112,113,114,115,116,106,117,119 };
        Eigen::Matrix<double, 113, 113> AM;
        for (int i = 0; i < 113; i++)
        {
            AM.row(i) = RR.row(AM_ind[i]);
        }

        Eigen::Matrix<std::complex<double>, 2, 113> sols;
        sols.setZero();

        // Solve eigenvalue problem
        double p[1+113];
        Eigen::Matrix<double, 113, 113> AMp = AM;
//        sturm::charpoly_danilevsky_piv(AMp, p);
        charpoly_danilevsky_piv(AMp, p);
        double roots[113];
        int nroots;
//        nroots = sturm::bisect_sturm<113>(p, roots, 1e-12);
        nroots = realRoots(p, roots, 113);

//        std::cout << "Roots: ";
//        for (int i = 0; i < nroots; ++i)
//            std::cout << roots[i];
//        std::cout << std::endl;

        relpose_kfEfk_fast_eigenvector_solver(roots, nroots, AM, sols);

        return sols;
    }

    int relpose_kfEfk(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                      std::vector<ImagePair> *models_){
        models_->clear();

        size_t kType = 2, kElim = 4;

        int sample_number_ = 7;
        Eigen::MatrixXd u(2, sample_number_), v(2, sample_number_);
        for (int i = 0; i < sample_number_; ++i)
        {
            u(0, i) = x1[i](0);
            u(1, i) = x1[i](1);
            v(0, i) = x2[i](0);
            v(1, i) = x2[i](1);
        }

        Eigen::MatrixXd A = Eigen::MatrixXd(sample_number_, 9), B = Eigen::MatrixXd(sample_number_, 9), C = Eigen::MatrixXd(sample_number_, 9), D = Eigen::MatrixXd(sample_number_, 9);

        if(!lincoeffs_k(u, v, kType, A, B, C, D))
        {
            std::cout << "Failed to compute the linear coefficients." << std::endl;
        }

        std::vector<int> ordo = {0, 1, 3, 4, 6, 7, 2, 5, 8};

        Eigen::MatrixXd Ar = Eigen::MatrixXd(sample_number_, 5), Br = Eigen::MatrixXd(sample_number_, 5), Cr = Eigen::MatrixXd(sample_number_, 5), Dr = Eigen::MatrixXd(sample_number_, 5);
        elimcoeffs_k(A, B, C, D, kElim, ordo, Ar, Br, Cr, Dr);

        Dr = Dr.rightCols(1);

        double data[167];
        relpose_kfEfk_prepare_data(Ar, Br, Dr, data);

        std::vector<int> sz = {50, 117};
        normalize_data_eqs(data, sz);

        Eigen::MatrixXcd sols = solver_new_kfEfk(data);

        std::vector<size_t> keptSolIndices;
        for(int i=0; i<sols.cols(); ++i)
        {
            if(sols.col(i).imag().norm() > 1e-10 || sols.col(i).real().norm() < 1e-10)
                continue;
            keptSolIndices.push_back(i);
        }
        Eigen::MatrixXd vsols(sols.rows(), keptSolIndices.size());
        for(size_t i=0; i<keptSolIndices.size(); ++i)
        {
            vsols.col(i) = sols.col(keptSolIndices[i]).real();
        }

        Eigen::VectorXd k(vsols.cols()), f(vsols.cols());
        k = vsols.row(0);
        f = vsols.row(1);

        Eigen::MatrixXd Fs = relpose_kfEfk_fundamental_from_sol(Ar, Br, Dr, k, f);

        for(int i=0; i<Fs.cols(); ++i)
        {
            if (f[i] < 0.0){
                continue;
            }

            double ff = f[i];
            Eigen::Matrix3d E;
            E <<    Fs.col(i)[0], Fs.col(i)[3], Fs.col(i)[6],
                    Fs.col(i)[1], Fs.col(i)[4], Fs.col(i)[7],
                    Fs.col(i)[2], Fs.col(i)[5], Fs.col(i)[8];

            Eigen::JacobiSVD<Eigen::Matrix3d, Eigen::ComputeThinU> svd(E);

            Eigen::DiagonalMatrix<double, 3> K(ff, ff, 1.0);
            E = K * E * K;
            CameraPoseVector poses;

            motion_from_essential(E, std::vector<Eigen::Vector3d>(), std::vector<Eigen::Vector3d>(), &poses);

            for (CameraPose pose : poses){
                Camera camera = Camera("DIVISION_RADIAL", std::vector<double>({ff, 0.0, 0.0, k[i]}), -1, -1);
                models_->push_back(ImagePair(pose, camera, camera));
            }
        }
        return models_->size();
    }
}