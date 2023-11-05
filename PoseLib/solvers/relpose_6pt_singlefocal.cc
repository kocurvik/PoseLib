#include "relpose_6pt_onefocal.h"

#include <Eigen/Dense>
#include <PoseLib/misc/essential.h>
#include <iostream>
#include <math.h>
#include <stdio.h>

#define RELERROR 1.0e-12 /* smallest relative error we want */
// #define MAXPOW        0        /* max power of 10 we wish to search to */
#define MAXIT 800 /* max number of iterations */
#define SMALL_ENOUGH                                                                                                   \
    1.0e-12 /* a coefficient smaller than SMALL_ENOUGH                                                                 \
/* is considered to be zero (0.0). */
#ifndef MAX_DEG
#define MAX_DEG 64
#endif

/* structure type for representing a polynomial */
typedef struct p {
    int ord;
    double coef[MAX_DEG + 1];
} poly;

/*---------------------------------------------------------------------------
 * evalpoly
 *
 * evaluate polynomial defined in coef returning its value.
 *--------------------------------------------------------------------------*/
namespace {

double evalpoly(int ord, double *coef, double x) {
    double *fp = &coef[ord];
    double f = *fp;

    for (fp--; fp >= coef; fp--)
        f = x * f + *fp;

    return (f);
}

int modrf_pos(int ord, double *coef, double a, double b, double *val, int invert) {
    int its;
    double fx, lfx;
    double *fp;
    double *scoef = coef;
    double *ecoef = &coef[ord];
    double fa, fb;

    // Invert the interval if required
    if (invert) {
        double temp = a;
        a = 1.0 / b;
        b = 1.0 / temp;
    }

    // Evaluate the polynomial at the end points
    if (invert) {
        fb = fa = *scoef;
        for (fp = scoef + 1; fp <= ecoef; fp++) {
            fa = a * fa + *fp;
            fb = b * fb + *fp;
        }
    } else {
        fb = fa = *ecoef;
        for (fp = ecoef - 1; fp >= scoef; fp--) {
            fa = a * fa + *fp;
            fb = b * fb + *fp;
        }
    }

    // if there is no sign difference the method won't work
    if (fa * fb > 0.0)
        return (0);

    // Return if the values are close to zero already
    if (fabs(fa) < RELERROR) {
        *val = invert ? 1.0 / a : a;
        return (1);
    }

    if (fabs(fb) < RELERROR) {
        *val = invert ? 1.0 / b : b;
        return (1);
    }

    lfx = fa;

    for (its = 0; its < MAXIT; its++) {
        // Assuming straight line from a to b, find zero
        double x = (fb * a - fa * b) / (fb - fa);

        // Evaluate the polynomial at x
        if (invert) {
            fx = *scoef;
            for (fp = scoef + 1; fp <= ecoef; fp++)
                fx = x * fx + *fp;
        } else {
            fx = *ecoef;
            for (fp = ecoef - 1; fp >= scoef; fp--)
                fx = x * fx + *fp;
        }

        // Evaluate two stopping conditions
        if (fabs(x) > RELERROR && fabs(fx / x) < RELERROR) {
            *val = invert ? 1.0 / x : x;
            return (1);
        } else if (fabs(fx) < RELERROR) {
            *val = invert ? 1.0 / x : x;
            return (1);
        }

        // Subdivide region, depending on whether fx has same sign as fa or fb
        if ((fa * fx) < 0) {
            b = x;
            fb = fx;
            if ((lfx * fx) > 0)
                fa /= 2;
        } else {
            a = x;
            fa = fx;
            if ((lfx * fx) > 0)
                fb /= 2;
        }

        // Return if the difference between a and b is very small
        if (fabs(b - a) < fabs(RELERROR * a)) {
            *val = invert ? 1.0 / a : a;
            return (1);
        }

        lfx = fx;
    }

    //==================================================================
    // This is debugging in case something goes wrong.
    // If we reach here, we have not converged -- give some diagnostics
    //==================================================================

    //fprintf(stderr, "modrf overflow on interval %f %f\n", a, b);
    //fprintf(stderr, "\t b-a = %12.5e\n", b - a);
    //fprintf(stderr, "\t fa  = %12.5e\n", fa);
    //fprintf(stderr, "\t fb  = %12.5e\n", fb);
    //fprintf(stderr, "\t fx  = %12.5e\n", fx);

    // Evaluate the true values at a and b
    if (invert) {
        fb = fa = *scoef;
        for (fp = scoef + 1; fp <= ecoef; fp++) {
            fa = a * fa + *fp;
            fb = b * fb + *fp;
        }
    } else {
        fb = fa = *ecoef;
        for (fp = ecoef - 1; fp >= scoef; fp--) {
            fa = a * fa + *fp;
            fb = b * fb + *fp;
        }
    }

    //fprintf(stderr, "\t true fa = %12.5e\n", fa);
    //fprintf(stderr, "\t true fb = %12.5e\n", fb);
    //fprintf(stderr, "\t gradient= %12.5e\n", (fb - fa) / (b - a));

    //// Print out the polynomial
    //fprintf(stderr, "Polynomial coefficients\n");
    //for (fp = ecoef; fp >= scoef; fp--)
    //    fprintf(stderr, "\t%12.5e\n", *fp);

    return (0);
}

/*---------------------------------------------------------------------------
 * modrf
 *
 * uses the modified regula-falsi method to evaluate the root
 * in interval [a,b] of the polynomial described in coef. The
 * root is returned is returned in *val. The routine returns zero
 * if it can't converge.
 *--------------------------------------------------------------------------*/

int modrf(int ord, double *coef, double a, double b, double *val) {
    // This is an interfact to modrf that takes account of different cases
    // The idea is that the basic routine works badly for polynomials on
    // intervals that extend well beyond [-1, 1], because numbers get too large

    double *fp;
    double *scoef = coef;
    double *ecoef = &coef[ord];
    const int invert = 1;

    double fp1 = 0.0, fm1 = 0.0; // Values of function at 1 and -1
    double fa = 0.0, fb = 0.0;   // Values at end points

    // We assume that a < b
    if (a > b) {
        double temp = a;
        a = b;
        b = temp;
    }

    // The normal case, interval is inside [-1, 1]
    if (b <= 1.0 && a >= -1.0)
        return modrf_pos(ord, coef, a, b, val, !invert);

    // The case where the interval is outside [-1, 1]
    if (a >= 1.0 || b <= -1.0)
        return modrf_pos(ord, coef, a, b, val, invert);

    // If we have got here, then the interval includes the points 1 or -1.
    // In this case, we need to evaluate at these points

    // Evaluate the polynomial at the end points
    for (fp = ecoef - 1; fp >= scoef; fp--) {
        fp1 = *fp + fp1;
        fm1 = *fp - fm1;
        fa = a * fa + *fp;
        fb = b * fb + *fp;
    }

    // Then there is the case where the interval contains -1 or 1
    if (a < -1.0 && b > 1.0) {
        // Interval crosses over 1.0, so cut
        if (fa * fm1 < 0.0) // The solution is between a and -1
            return modrf_pos(ord, coef, a, -1.0, val, invert);
        else if (fb * fp1 < 0.0) // The solution is between 1 and b
            return modrf_pos(ord, coef, 1.0, b, val, invert);
        else // The solution is between -1 and 1
            return modrf_pos(ord, coef, -1.0, 1.0, val, !invert);
    } else if (a < -1.0) {
        // Interval crosses over 1.0, so cut
        if (fa * fm1 < 0.0) // The solution is between a and -1
            return modrf_pos(ord, coef, a, -1.0, val, invert);
        else // The solution is between -1 and b
            return modrf_pos(ord, coef, -1.0, b, val, !invert);
    } else // b > 1.0
    {
        if (fb * fp1 < 0.0) // The solution is between 1 and b
            return modrf_pos(ord, coef, 1.0, b, val, invert);
        else // The solution is between a and 1
            return modrf_pos(ord, coef, a, 1.0, val, !invert);
    }
}

/*---------------------------------------------------------------------------
 * modp
 *
 *  calculates the modulus of u(x) / v(x) leaving it in r, it
 *  returns 0 if r(x) is a constant.
 *  note: this function assumes the leading coefficient of v is 1 or -1
 *--------------------------------------------------------------------------*/

static int modp(poly *u, poly *v, poly *r) {
    int j, k; /* Loop indices */

    double *nr = r->coef;
    double *end = &u->coef[u->ord];

    double *uc = u->coef;
    while (uc <= end)
        *nr++ = *uc++;

    if (v->coef[v->ord] < 0.0) {

        for (k = u->ord - v->ord - 1; k >= 0; k -= 2)
            r->coef[k] = -r->coef[k];

        for (k = u->ord - v->ord; k >= 0; k--)
            for (j = v->ord + k - 1; j >= k; j--)
                r->coef[j] = -r->coef[j] - r->coef[v->ord + k] * v->coef[j - k];
    } else {
        for (k = u->ord - v->ord; k >= 0; k--)
            for (j = v->ord + k - 1; j >= k; j--)
                r->coef[j] -= r->coef[v->ord + k] * v->coef[j - k];
    }

    k = v->ord - 1;
    while (k >= 0 && fabs(r->coef[k]) < SMALL_ENOUGH) {
        r->coef[k] = 0.0;
        k--;
    }

    r->ord = (k < 0) ? 0 : k;

    return (r->ord);
}

/*---------------------------------------------------------------------------
 * buildsturm
 *
 * build up a sturm sequence for a polynomial in smat, returning
 * the number of polynomials in the sequence
 *--------------------------------------------------------------------------*/

int buildsturm(int ord, poly *sseq) {
    sseq[0].ord = ord;
    sseq[1].ord = ord - 1;

    /* calculate the derivative and normalise the leading coefficient */
    {
        int i; // Loop index
        poly *sp;
        double f = fabs(sseq[0].coef[ord] * ord);
        double *fp = sseq[1].coef;
        double *fc = sseq[0].coef + 1;

        for (i = 1; i <= ord; i++)
            *fp++ = *fc++ * i / f;

        /* construct the rest of the Sturm sequence */
        for (sp = sseq + 2; modp(sp - 2, sp - 1, sp); sp++) {

            /* reverse the sign and normalise */
            f = -fabs(sp->coef[sp->ord]);
            for (fp = &sp->coef[sp->ord]; fp >= sp->coef; fp--)
                *fp /= f;
        }

        sp->coef[0] = -sp->coef[0]; /* reverse the sign */

        return (sp - sseq);
    }
}

/*---------------------------------------------------------------------------
 * numchanges
 *
 * return the number of sign changes in the Sturm sequence in
 * sseq at the value a.
 *--------------------------------------------------------------------------*/

int numchanges(int np, poly *sseq, double a) {
    int changes = 0;

    double lf = evalpoly(sseq[0].ord, sseq[0].coef, a);

    poly *s;
    for (s = sseq + 1; s <= sseq + np; s++) {
        double f = evalpoly(s->ord, s->coef, a);
        if (lf == 0.0 || lf * f < 0)
            changes++;
        lf = f;
    }

    return (changes);
}

/*---------------------------------------------------------------------------
 * numroots
 *
 * return the number of distinct real roots of the polynomial described in sseq.
 *--------------------------------------------------------------------------*/

int numroots(int np, poly *sseq, int *atneg, int *atpos, bool non_neg) {
    int atposinf = 0;
    int atneginf = 0;

    /* changes at positive infinity */
    double f;
    double lf = sseq[0].coef[sseq[0].ord];

    poly *s;
    for (s = sseq + 1; s <= sseq + np; s++) {
        f = s->coef[s->ord];
        if (lf == 0.0 || lf * f < 0)
            atposinf++;
        lf = f;
    }

    // changes at negative infinity or zero
    if (non_neg)
        atneginf = numchanges(np, sseq, 0.0);

    else {
        if (sseq[0].ord & 1)
            lf = -sseq[0].coef[sseq[0].ord];
        else
            lf = sseq[0].coef[sseq[0].ord];

        for (s = sseq + 1; s <= sseq + np; s++) {
            if (s->ord & 1)
                f = -s->coef[s->ord];
            else
                f = s->coef[s->ord];
            if (lf == 0.0 || lf * f < 0)
                atneginf++;
            lf = f;
        }
    }

    *atneg = atneginf;
    *atpos = atposinf;

    return (atneginf - atposinf);
}

/*---------------------------------------------------------------------------
 * sbisect
 *
 * uses a bisection based on the sturm sequence for the polynomial
 * described in sseq to isolate intervals in which roots occur,
 * the roots are returned in the roots array in order of magnitude.
 *--------------------------------------------------------------------------*/

int sbisect(int np, poly *sseq, double min, double max, int atmin, int atmax, double *roots) {
    double mid;
    int atmid;
    int its;
    int n1 = 0, n2 = 0;
    int nroot = atmin - atmax;

    if (nroot == 1) {

        /* first try a less expensive technique.  */
        if (modrf(sseq->ord, sseq->coef, min, max, &roots[0]))
            return 1;

        /*
         * if we get here we have to evaluate the root the hard
         * way by using the Sturm sequence.
         */
        for (its = 0; its < MAXIT; its++) {
            mid = (double)((min + max) / 2);
            atmid = numchanges(np, sseq, mid);

            if (fabs(mid) > RELERROR) {
                if (fabs((max - min) / mid) < RELERROR) {
                    roots[0] = mid;
                    return 1;
                }
            } else if (fabs(max - min) < RELERROR) {
                roots[0] = mid;
                return 1;
            }

            if ((atmin - atmid) == 0)
                min = mid;
            else
                max = mid;
        }

        if (its == MAXIT) {
            /*fprintf(stderr, "sbisect: overflow min %f max %f\
							                         diff %e nroot %d n1 %d n2 %d\n",
                    min, max, max - min, nroot, n1, n2);*/
            roots[0] = mid;
        }

        return 1;
    }

    /* more than one root in the interval, we have to bisect */
    for (its = 0; its < MAXIT; its++) {

        mid = (double)((min + max) / 2);
        atmid = numchanges(np, sseq, mid);

        n1 = atmin - atmid;
        n2 = atmid - atmax;

        if (n1 != 0 && n2 != 0) {
            sbisect(np, sseq, min, mid, atmin, atmid, roots);
            sbisect(np, sseq, mid, max, atmid, atmax, &roots[n1]);
            break;
        }

        if (n1 == 0)
            min = mid;
        else
            max = mid;
    }

    if (its == MAXIT) {
        /*fprintf(stderr, "sbisect: roots too close together\n");
        fprintf(stderr, "sbisect: overflow min %f max %f diff %e\
						                      nroot %d n1 %d n2 %d\n",
                min, max, max - min, nroot, n1, n2);*/
        for (n1 = atmax; n1 < atmin; n1++)
            roots[n1 - atmax] = mid;
    }

    return 1;
}

int find_real_roots_sturm(double *p, int order, double *roots, int *nroots, int maxpow, bool non_neg) {
    /*
     * finds the roots of the input polynomial.  They are returned in roots.
     * It is assumed that roots is already allocated with space for the roots.
     */

    poly sseq[MAX_DEG + 1];
    double min, max;
    int i, nchanges, np, atmin, atmax;

    // Copy the coefficients from the input p.  Normalize as we go
    double norm = 1.0 / p[order];
    for (i = 0; i <= order; i++)
        sseq[0].coef[i] = p[i] * norm;

    // Now, also normalize the other terms
    double val0 = fabs(sseq[0].coef[0]);
    double fac = 1.0; // This will be a factor for the roots
    if (val0 > 10.0)  // Do this in case there are zero roots
    {
        fac = pow(val0, -1.0 / order);
        double mult = fac;
        for (int i = order - 1; i >= 0; i--) {
            sseq[0].coef[i] *= mult;
            mult = mult * fac;
        }
    }

    /* build the Sturm sequence */
    np = buildsturm(order, sseq);

#ifdef RH_DEBUG
    {
        int i, j;

        printf("Sturm sequence for:\n");
        for (i = order; i >= 0; i--)
            printf("%lf ", sseq[0].coef[i]);
        printf("\n\n");

        for (i = 0; i <= np; i++) {
            for (j = sseq[i].ord; j >= 0; j--)
                printf("%10f ", sseq[i].coef[j]);
            printf("\n");
        }

        printf("\n");
    }
#endif

    // get the number of real roots
    *nroots = numroots(np, sseq, &atmin, &atmax, non_neg);

    if (*nroots == 0) {
        // fprintf(stderr, "solve: no real roots\n");
        return 0;
    }

    /* calculate the bracket that the roots live in */
    if (non_neg)
        min = 0.0;
    else {
        min = -1.0;
        nchanges = numchanges(np, sseq, min);
        for (i = 0; nchanges != atmin && i != maxpow; i++) {
            min *= 10.0;
            nchanges = numchanges(np, sseq, min);
        }

        if (nchanges != atmin) {
            // printf("solve: unable to bracket all negative roots\n");
            atmin = nchanges;
        }
    }

    max = 1.0;
    nchanges = numchanges(np, sseq, max);
    for (i = 0; nchanges != atmax && i != maxpow; i++) {
        max *= 10.0;
        nchanges = numchanges(np, sseq, max);
    }

    if (nchanges != atmax) {
        // printf("solve: unable to bracket all positive roots\n");
        atmax = nchanges;
    }

    *nroots = atmin - atmax;

    /* perform the bisection */
    sbisect(np, sseq, min, max, atmin, atmax, roots);

    /* Finally, reorder the roots */
    for (i = 0; i < *nroots; i++)
        roots[i] /= fac;

#ifdef RH_DEBUG

    /* write out the roots */
    printf("Number of roots = %d\n", *nroots);
    for (i = 0; i < *nroots; i++)
        printf("%12.5e\n", roots[i]);

#endif

    return 1;
}
} // namespace

template <typename Derived> void charpoly_danilevsky_piv(Eigen::MatrixBase<Derived> &A, double *p) {
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
        A.row(i - 1) = v.transpose() * A;

        Eigen::VectorXd vinv = (-1.0) * v;
        vinv(i - 1) = 1;
        vinv /= piv;
        vinv(i - 1) -= 1;
        Eigen::VectorXd Acol = A.col(i - 1);
        for (int j = 0; j <= i; j++)
            A.row(j) = A.row(j) + Acol(j) * vinv.transpose();

        A.row(i) = Eigen::VectorXd::Zero(n);
        A(i, i - 1) = 1;
    }
    p[n] = 1;
    for (int i = 0; i < n; i++)
        p[i] = -A(0, n - i - 1);
}

void fast_eigenvector_solver(double *eigv, int neig, Eigen::Matrix<double, 15, 15> &AM,
                             Eigen::Matrix<std::complex<double>, 3, 15> &sols) {
    static const int ind[] = {2, 3, 4, 6, 8, 9, 11, 14};
    // Truncated action matrix containing non-trivial rows
    Eigen::Matrix<double, 8, 15> AMs;
    double zi[3];

    for (int i = 0; i < 8; i++) {
        AMs.row(i) = AM.row(ind[i]);
    }
    for (int i = 0; i < neig; i++) {
        zi[0] = eigv[i];
        for (int j = 1; j < 3; j++) {
            zi[j] = zi[j - 1] * eigv[i];
        }
        Eigen::Matrix<double, 8, 8> AA;
        AA.col(0) = AMs.col(2);
        AA.col(1) = AMs.col(6);
        AA.col(2) = zi[0] * AMs.col(4) + AMs.col(5);
        AA.col(3) = AMs.col(1) + zi[0] * AMs.col(3);
        AA.col(4) = AMs.col(14);
        AA.col(5) = zi[0] * AMs.col(11) + AMs.col(13);
        AA.col(6) = zi[1] * AMs.col(9) + zi[0] * AMs.col(10) + AMs.col(12);
        AA.col(7) = AMs.col(0) + zi[0] * AMs.col(7) + zi[1] * AMs.col(8);
        AA(0, 0) = AA(0, 0) - zi[0];
        AA(3, 1) = AA(3, 1) - zi[0];
        AA(2, 2) = AA(2, 2) - zi[1];
        AA(1, 3) = AA(1, 3) - zi[1];
        AA(7, 4) = AA(7, 4) - zi[0];
        AA(6, 5) = AA(6, 5) - zi[1];
        AA(5, 6) = AA(5, 6) - zi[2];
        AA(4, 7) = AA(4, 7) - zi[2];

        Eigen::Matrix<double, 7, 1> s = AA.leftCols(7).colPivHouseholderQr().solve(-AA.col(7));
        sols(0, i) = s(3);
        sols(1, i) = zi[0];
        sols(2, i) = s(6);
    }
}


int solver_relpose_6pt_singlefocal(const Eigen::VectorXd &data, Eigen::Matrix<std::complex<double>, 3, 15> &sols) {
    // Compute coefficients
    const double *d = data.data();
    Eigen::VectorXd coeffs(280);
    coeffs[0] = 2 * d[11] * d[15] * d[17] - d[9] * std::pow(d[17], 2);
    coeffs[1] = -std::pow(d[17], 2) * d[18] + 2 * d[15] * d[17] * d[20] + 2 * d[11] * d[17] * d[24] +
                2 * d[11] * d[15] * d[26] - 2 * d[9] * d[17] * d[26];
    coeffs[2] = 2 * d[17] * d[20] * d[24] - 2 * d[17] * d[18] * d[26] + 2 * d[15] * d[20] * d[26] +
                2 * d[11] * d[24] * d[26] - d[9] * std::pow(d[26], 2);
    coeffs[3] = 2 * d[20] * d[24] * d[26] - d[18] * std::pow(d[26], 2);
    coeffs[4] = d[9] * std::pow(d[11], 2) + 2 * d[11] * d[12] * d[14] - d[9] * std::pow(d[14], 2) +
                d[9] * std::pow(d[15], 2) + 2 * d[10] * d[15] * d[16] - d[9] * std::pow(d[16], 2);
    coeffs[5] = std::pow(d[11], 2) * d[18] - std::pow(d[14], 2) * d[18] + std::pow(d[15], 2) * d[18] -
                std::pow(d[16], 2) * d[18] + 2 * d[15] * d[16] * d[19] + 2 * d[9] * d[11] * d[20] +
                2 * d[12] * d[14] * d[20] + 2 * d[11] * d[14] * d[21] + 2 * d[11] * d[12] * d[23] -
                2 * d[9] * d[14] * d[23] + 2 * d[9] * d[15] * d[24] + 2 * d[10] * d[16] * d[24] +
                2 * d[10] * d[15] * d[25] - 2 * d[9] * d[16] * d[25];
    coeffs[6] = 2 * d[11] * d[18] * d[20] + d[9] * std::pow(d[20], 2) + 2 * d[14] * d[20] * d[21] -
                2 * d[14] * d[18] * d[23] + 2 * d[12] * d[20] * d[23] + 2 * d[11] * d[21] * d[23] -
                d[9] * std::pow(d[23], 2) + 2 * d[15] * d[18] * d[24] + 2 * d[16] * d[19] * d[24] +
                d[9] * std::pow(d[24], 2) - 2 * d[16] * d[18] * d[25] + 2 * d[15] * d[19] * d[25] +
                2 * d[10] * d[24] * d[25] - d[9] * std::pow(d[25], 2);
    coeffs[7] = d[18] * std::pow(d[20], 2) + 2 * d[20] * d[21] * d[23] - d[18] * std::pow(d[23], 2) +
                d[18] * std::pow(d[24], 2) + 2 * d[19] * d[24] * d[25] - d[18] * std::pow(d[25], 2);
    coeffs[8] = 2 * d[8] * d[11] * d[15] - 2 * d[8] * d[9] * d[17] + 2 * d[6] * d[11] * d[17] +
                2 * d[2] * d[15] * d[17] - d[0] * std::pow(d[17], 2);
    coeffs[9] = -2 * d[8] * d[17] * d[18] + 2 * d[8] * d[15] * d[20] + 2 * d[6] * d[17] * d[20] +
                2 * d[8] * d[11] * d[24] + 2 * d[2] * d[17] * d[24] - 2 * d[8] * d[9] * d[26] +
                2 * d[6] * d[11] * d[26] + 2 * d[2] * d[15] * d[26] - 2 * d[0] * d[17] * d[26];
    coeffs[10] = 2 * d[8] * d[20] * d[24] - 2 * d[8] * d[18] * d[26] + 2 * d[6] * d[20] * d[26] +
                 2 * d[2] * d[24] * d[26] - d[0] * std::pow(d[26], 2);
    coeffs[11] = std::pow(d[9], 3) + d[9] * std::pow(d[10], 2) + d[9] * std::pow(d[12], 2) + 2 * d[10] * d[12] * d[13] -
                 d[9] * std::pow(d[13], 2);
    coeffs[12] = 3 * std::pow(d[9], 2) * d[18] + std::pow(d[10], 2) * d[18] + std::pow(d[12], 2) * d[18] -
                 std::pow(d[13], 2) * d[18] + 2 * d[9] * d[10] * d[19] + 2 * d[12] * d[13] * d[19] +
                 2 * d[9] * d[12] * d[21] + 2 * d[10] * d[13] * d[21] + 2 * d[10] * d[12] * d[22] -
                 2 * d[9] * d[13] * d[22];
    coeffs[13] = 3 * d[9] * std::pow(d[18], 2) + 2 * d[10] * d[18] * d[19] + d[9] * std::pow(d[19], 2) +
                 2 * d[12] * d[18] * d[21] + 2 * d[13] * d[19] * d[21] + d[9] * std::pow(d[21], 2) -
                 2 * d[13] * d[18] * d[22] + 2 * d[12] * d[19] * d[22] + 2 * d[10] * d[21] * d[22] -
                 d[9] * std::pow(d[22], 2);
    coeffs[14] = std::pow(d[18], 3) + d[18] * std::pow(d[19], 2) + d[18] * std::pow(d[21], 2) +
                 2 * d[19] * d[21] * d[22] - d[18] * std::pow(d[22], 2);
    coeffs[15] = 2 * d[2] * d[9] * d[11] + d[0] * std::pow(d[11], 2) + 2 * d[5] * d[11] * d[12] -
                 2 * d[5] * d[9] * d[14] + 2 * d[3] * d[11] * d[14] + 2 * d[2] * d[12] * d[14] -
                 d[0] * std::pow(d[14], 2) + 2 * d[6] * d[9] * d[15] + 2 * d[7] * d[10] * d[15] +
                 d[0] * std::pow(d[15], 2) - 2 * d[7] * d[9] * d[16] + 2 * d[6] * d[10] * d[16] +
                 2 * d[1] * d[15] * d[16] - d[0] * std::pow(d[16], 2);
    coeffs[16] =
        2 * d[2] * d[11] * d[18] - 2 * d[5] * d[14] * d[18] + 2 * d[6] * d[15] * d[18] - 2 * d[7] * d[16] * d[18] +
        2 * d[7] * d[15] * d[19] + 2 * d[6] * d[16] * d[19] + 2 * d[2] * d[9] * d[20] + 2 * d[0] * d[11] * d[20] +
        2 * d[5] * d[12] * d[20] + 2 * d[3] * d[14] * d[20] + 2 * d[5] * d[11] * d[21] + 2 * d[2] * d[14] * d[21] -
        2 * d[5] * d[9] * d[23] + 2 * d[3] * d[11] * d[23] + 2 * d[2] * d[12] * d[23] - 2 * d[0] * d[14] * d[23] +
        2 * d[6] * d[9] * d[24] + 2 * d[7] * d[10] * d[24] + 2 * d[0] * d[15] * d[24] + 2 * d[1] * d[16] * d[24] -
        2 * d[7] * d[9] * d[25] + 2 * d[6] * d[10] * d[25] + 2 * d[1] * d[15] * d[25] - 2 * d[0] * d[16] * d[25];
    coeffs[17] = 2 * d[2] * d[18] * d[20] + d[0] * std::pow(d[20], 2) + 2 * d[5] * d[20] * d[21] -
                 2 * d[5] * d[18] * d[23] + 2 * d[3] * d[20] * d[23] + 2 * d[2] * d[21] * d[23] -
                 d[0] * std::pow(d[23], 2) + 2 * d[6] * d[18] * d[24] + 2 * d[7] * d[19] * d[24] +
                 d[0] * std::pow(d[24], 2) - 2 * d[7] * d[18] * d[25] + 2 * d[6] * d[19] * d[25] +
                 2 * d[1] * d[24] * d[25] - d[0] * std::pow(d[25], 2);
    coeffs[18] = -std::pow(d[8], 2) * d[9] + 2 * d[6] * d[8] * d[11] + 2 * d[2] * d[8] * d[15] +
                 2 * d[2] * d[6] * d[17] - 2 * d[0] * d[8] * d[17];
    coeffs[19] = -std::pow(d[8], 2) * d[18] + 2 * d[6] * d[8] * d[20] + 2 * d[2] * d[8] * d[24] +
                 2 * d[2] * d[6] * d[26] - 2 * d[0] * d[8] * d[26];
    coeffs[20] = 3 * d[0] * std::pow(d[9], 2) + 2 * d[1] * d[9] * d[10] + d[0] * std::pow(d[10], 2) +
                 2 * d[3] * d[9] * d[12] + 2 * d[4] * d[10] * d[12] + d[0] * std::pow(d[12], 2) -
                 2 * d[4] * d[9] * d[13] + 2 * d[3] * d[10] * d[13] + 2 * d[1] * d[12] * d[13] -
                 d[0] * std::pow(d[13], 2);
    coeffs[21] =
        6 * d[0] * d[9] * d[18] + 2 * d[1] * d[10] * d[18] + 2 * d[3] * d[12] * d[18] - 2 * d[4] * d[13] * d[18] +
        2 * d[1] * d[9] * d[19] + 2 * d[0] * d[10] * d[19] + 2 * d[4] * d[12] * d[19] + 2 * d[3] * d[13] * d[19] +
        2 * d[3] * d[9] * d[21] + 2 * d[4] * d[10] * d[21] + 2 * d[0] * d[12] * d[21] + 2 * d[1] * d[13] * d[21] -
        2 * d[4] * d[9] * d[22] + 2 * d[3] * d[10] * d[22] + 2 * d[1] * d[12] * d[22] - 2 * d[0] * d[13] * d[22];
    coeffs[22] = 3 * d[0] * std::pow(d[18], 2) + 2 * d[1] * d[18] * d[19] + d[0] * std::pow(d[19], 2) +
                 2 * d[3] * d[18] * d[21] + 2 * d[4] * d[19] * d[21] + d[0] * std::pow(d[21], 2) -
                 2 * d[4] * d[18] * d[22] + 2 * d[3] * d[19] * d[22] + 2 * d[1] * d[21] * d[22] -
                 d[0] * std::pow(d[22], 2);
    coeffs[23] = std::pow(d[2], 2) * d[9] - std::pow(d[5], 2) * d[9] + std::pow(d[6], 2) * d[9] -
                 std::pow(d[7], 2) * d[9] + 2 * d[6] * d[7] * d[10] + 2 * d[0] * d[2] * d[11] +
                 2 * d[3] * d[5] * d[11] + 2 * d[2] * d[5] * d[12] + 2 * d[2] * d[3] * d[14] - 2 * d[0] * d[5] * d[14] +
                 2 * d[0] * d[6] * d[15] + 2 * d[1] * d[7] * d[15] + 2 * d[1] * d[6] * d[16] - 2 * d[0] * d[7] * d[16];
    coeffs[24] = std::pow(d[2], 2) * d[18] - std::pow(d[5], 2) * d[18] + std::pow(d[6], 2) * d[18] -
                 std::pow(d[7], 2) * d[18] + 2 * d[6] * d[7] * d[19] + 2 * d[0] * d[2] * d[20] +
                 2 * d[3] * d[5] * d[20] + 2 * d[2] * d[5] * d[21] + 2 * d[2] * d[3] * d[23] - 2 * d[0] * d[5] * d[23] +
                 2 * d[0] * d[6] * d[24] + 2 * d[1] * d[7] * d[24] + 2 * d[1] * d[6] * d[25] - 2 * d[0] * d[7] * d[25];
    coeffs[25] = 2 * d[2] * d[6] * d[8] - d[0] * std::pow(d[8], 2);
    coeffs[26] = 3 * std::pow(d[0], 2) * d[9] + std::pow(d[1], 2) * d[9] + std::pow(d[3], 2) * d[9] -
                 std::pow(d[4], 2) * d[9] + 2 * d[0] * d[1] * d[10] + 2 * d[3] * d[4] * d[10] +
                 2 * d[0] * d[3] * d[12] + 2 * d[1] * d[4] * d[12] + 2 * d[1] * d[3] * d[13] - 2 * d[0] * d[4] * d[13];
    coeffs[27] = 3 * std::pow(d[0], 2) * d[18] + std::pow(d[1], 2) * d[18] + std::pow(d[3], 2) * d[18] -
                 std::pow(d[4], 2) * d[18] + 2 * d[0] * d[1] * d[19] + 2 * d[3] * d[4] * d[19] +
                 2 * d[0] * d[3] * d[21] + 2 * d[1] * d[4] * d[21] + 2 * d[1] * d[3] * d[22] - 2 * d[0] * d[4] * d[22];
    coeffs[28] = d[0] * std::pow(d[2], 2) + 2 * d[2] * d[3] * d[5] - d[0] * std::pow(d[5], 2) +
                 d[0] * std::pow(d[6], 2) + 2 * d[1] * d[6] * d[7] - d[0] * std::pow(d[7], 2);
    coeffs[29] = std::pow(d[0], 3) + d[0] * std::pow(d[1], 2) + d[0] * std::pow(d[3], 2) + 2 * d[1] * d[3] * d[4] -
                 d[0] * std::pow(d[4], 2);
    coeffs[30] = 2 * d[11] * d[16] * d[17] - d[10] * std::pow(d[17], 2);
    coeffs[31] = -std::pow(d[17], 2) * d[19] + 2 * d[16] * d[17] * d[20] + 2 * d[11] * d[17] * d[25] +
                 2 * d[11] * d[16] * d[26] - 2 * d[10] * d[17] * d[26];
    coeffs[32] = 2 * d[17] * d[20] * d[25] - 2 * d[17] * d[19] * d[26] + 2 * d[16] * d[20] * d[26] +
                 2 * d[11] * d[25] * d[26] - d[10] * std::pow(d[26], 2);
    coeffs[33] = 2 * d[20] * d[25] * d[26] - d[19] * std::pow(d[26], 2);
    coeffs[34] = d[10] * std::pow(d[11], 2) + 2 * d[11] * d[13] * d[14] - d[10] * std::pow(d[14], 2) -
                 d[10] * std::pow(d[15], 2) + 2 * d[9] * d[15] * d[16] + d[10] * std::pow(d[16], 2);
    coeffs[35] = 2 * d[15] * d[16] * d[18] + std::pow(d[11], 2) * d[19] - std::pow(d[14], 2) * d[19] -
                 std::pow(d[15], 2) * d[19] + std::pow(d[16], 2) * d[19] + 2 * d[10] * d[11] * d[20] +
                 2 * d[13] * d[14] * d[20] + 2 * d[11] * d[14] * d[22] + 2 * d[11] * d[13] * d[23] -
                 2 * d[10] * d[14] * d[23] - 2 * d[10] * d[15] * d[24] + 2 * d[9] * d[16] * d[24] +
                 2 * d[9] * d[15] * d[25] + 2 * d[10] * d[16] * d[25];
    coeffs[36] = 2 * d[11] * d[19] * d[20] + d[10] * std::pow(d[20], 2) + 2 * d[14] * d[20] * d[22] -
                 2 * d[14] * d[19] * d[23] + 2 * d[13] * d[20] * d[23] + 2 * d[11] * d[22] * d[23] -
                 d[10] * std::pow(d[23], 2) + 2 * d[16] * d[18] * d[24] - 2 * d[15] * d[19] * d[24] -
                 d[10] * std::pow(d[24], 2) + 2 * d[15] * d[18] * d[25] + 2 * d[16] * d[19] * d[25] +
                 2 * d[9] * d[24] * d[25] + d[10] * std::pow(d[25], 2);
    coeffs[37] = d[19] * std::pow(d[20], 2) + 2 * d[20] * d[22] * d[23] - d[19] * std::pow(d[23], 2) -
                 d[19] * std::pow(d[24], 2) + 2 * d[18] * d[24] * d[25] + d[19] * std::pow(d[25], 2);
    coeffs[38] = 2 * d[8] * d[11] * d[16] - 2 * d[8] * d[10] * d[17] + 2 * d[7] * d[11] * d[17] +
                 2 * d[2] * d[16] * d[17] - d[1] * std::pow(d[17], 2);
    coeffs[39] = -2 * d[8] * d[17] * d[19] + 2 * d[8] * d[16] * d[20] + 2 * d[7] * d[17] * d[20] +
                 2 * d[8] * d[11] * d[25] + 2 * d[2] * d[17] * d[25] - 2 * d[8] * d[10] * d[26] +
                 2 * d[7] * d[11] * d[26] + 2 * d[2] * d[16] * d[26] - 2 * d[1] * d[17] * d[26];
    coeffs[40] = 2 * d[8] * d[20] * d[25] - 2 * d[8] * d[19] * d[26] + 2 * d[7] * d[20] * d[26] +
                 2 * d[2] * d[25] * d[26] - d[1] * std::pow(d[26], 2);
    coeffs[41] = std::pow(d[9], 2) * d[10] + std::pow(d[10], 3) - d[10] * std::pow(d[12], 2) +
                 2 * d[9] * d[12] * d[13] + d[10] * std::pow(d[13], 2);
    coeffs[42] = 2 * d[9] * d[10] * d[18] + 2 * d[12] * d[13] * d[18] + std::pow(d[9], 2) * d[19] +
                 3 * std::pow(d[10], 2) * d[19] - std::pow(d[12], 2) * d[19] + std::pow(d[13], 2) * d[19] -
                 2 * d[10] * d[12] * d[21] + 2 * d[9] * d[13] * d[21] + 2 * d[9] * d[12] * d[22] +
                 2 * d[10] * d[13] * d[22];
    coeffs[43] = d[10] * std::pow(d[18], 2) + 2 * d[9] * d[18] * d[19] + 3 * d[10] * std::pow(d[19], 2) +
                 2 * d[13] * d[18] * d[21] - 2 * d[12] * d[19] * d[21] - d[10] * std::pow(d[21], 2) +
                 2 * d[12] * d[18] * d[22] + 2 * d[13] * d[19] * d[22] + 2 * d[9] * d[21] * d[22] +
                 d[10] * std::pow(d[22], 2);
    coeffs[44] = std::pow(d[18], 2) * d[19] + std::pow(d[19], 3) - d[19] * std::pow(d[21], 2) +
                 2 * d[18] * d[21] * d[22] + d[19] * std::pow(d[22], 2);
    coeffs[45] = 2 * d[2] * d[10] * d[11] + d[1] * std::pow(d[11], 2) + 2 * d[5] * d[11] * d[13] -
                 2 * d[5] * d[10] * d[14] + 2 * d[4] * d[11] * d[14] + 2 * d[2] * d[13] * d[14] -
                 d[1] * std::pow(d[14], 2) + 2 * d[7] * d[9] * d[15] - 2 * d[6] * d[10] * d[15] -
                 d[1] * std::pow(d[15], 2) + 2 * d[6] * d[9] * d[16] + 2 * d[7] * d[10] * d[16] +
                 2 * d[0] * d[15] * d[16] + d[1] * std::pow(d[16], 2);
    coeffs[46] =
        2 * d[7] * d[15] * d[18] + 2 * d[6] * d[16] * d[18] + 2 * d[2] * d[11] * d[19] - 2 * d[5] * d[14] * d[19] -
        2 * d[6] * d[15] * d[19] + 2 * d[7] * d[16] * d[19] + 2 * d[2] * d[10] * d[20] + 2 * d[1] * d[11] * d[20] +
        2 * d[5] * d[13] * d[20] + 2 * d[4] * d[14] * d[20] + 2 * d[5] * d[11] * d[22] + 2 * d[2] * d[14] * d[22] -
        2 * d[5] * d[10] * d[23] + 2 * d[4] * d[11] * d[23] + 2 * d[2] * d[13] * d[23] - 2 * d[1] * d[14] * d[23] +
        2 * d[7] * d[9] * d[24] - 2 * d[6] * d[10] * d[24] - 2 * d[1] * d[15] * d[24] + 2 * d[0] * d[16] * d[24] +
        2 * d[6] * d[9] * d[25] + 2 * d[7] * d[10] * d[25] + 2 * d[0] * d[15] * d[25] + 2 * d[1] * d[16] * d[25];
    coeffs[47] = 2 * d[2] * d[19] * d[20] + d[1] * std::pow(d[20], 2) + 2 * d[5] * d[20] * d[22] -
                 2 * d[5] * d[19] * d[23] + 2 * d[4] * d[20] * d[23] + 2 * d[2] * d[22] * d[23] -
                 d[1] * std::pow(d[23], 2) + 2 * d[7] * d[18] * d[24] - 2 * d[6] * d[19] * d[24] -
                 d[1] * std::pow(d[24], 2) + 2 * d[6] * d[18] * d[25] + 2 * d[7] * d[19] * d[25] +
                 2 * d[0] * d[24] * d[25] + d[1] * std::pow(d[25], 2);
    coeffs[48] = -std::pow(d[8], 2) * d[10] + 2 * d[7] * d[8] * d[11] + 2 * d[2] * d[8] * d[16] +
                 2 * d[2] * d[7] * d[17] - 2 * d[1] * d[8] * d[17];
    coeffs[49] = -std::pow(d[8], 2) * d[19] + 2 * d[7] * d[8] * d[20] + 2 * d[2] * d[8] * d[25] +
                 2 * d[2] * d[7] * d[26] - 2 * d[1] * d[8] * d[26];
    coeffs[50] = d[1] * std::pow(d[9], 2) + 2 * d[0] * d[9] * d[10] + 3 * d[1] * std::pow(d[10], 2) +
                 2 * d[4] * d[9] * d[12] - 2 * d[3] * d[10] * d[12] - d[1] * std::pow(d[12], 2) +
                 2 * d[3] * d[9] * d[13] + 2 * d[4] * d[10] * d[13] + 2 * d[0] * d[12] * d[13] +
                 d[1] * std::pow(d[13], 2);
    coeffs[51] =
        2 * d[1] * d[9] * d[18] + 2 * d[0] * d[10] * d[18] + 2 * d[4] * d[12] * d[18] + 2 * d[3] * d[13] * d[18] +
        2 * d[0] * d[9] * d[19] + 6 * d[1] * d[10] * d[19] - 2 * d[3] * d[12] * d[19] + 2 * d[4] * d[13] * d[19] +
        2 * d[4] * d[9] * d[21] - 2 * d[3] * d[10] * d[21] - 2 * d[1] * d[12] * d[21] + 2 * d[0] * d[13] * d[21] +
        2 * d[3] * d[9] * d[22] + 2 * d[4] * d[10] * d[22] + 2 * d[0] * d[12] * d[22] + 2 * d[1] * d[13] * d[22];
    coeffs[52] = d[1] * std::pow(d[18], 2) + 2 * d[0] * d[18] * d[19] + 3 * d[1] * std::pow(d[19], 2) +
                 2 * d[4] * d[18] * d[21] - 2 * d[3] * d[19] * d[21] - d[1] * std::pow(d[21], 2) +
                 2 * d[3] * d[18] * d[22] + 2 * d[4] * d[19] * d[22] + 2 * d[0] * d[21] * d[22] +
                 d[1] * std::pow(d[22], 2);
    coeffs[53] = 2 * d[6] * d[7] * d[9] + std::pow(d[2], 2) * d[10] - std::pow(d[5], 2) * d[10] -
                 std::pow(d[6], 2) * d[10] + std::pow(d[7], 2) * d[10] + 2 * d[1] * d[2] * d[11] +
                 2 * d[4] * d[5] * d[11] + 2 * d[2] * d[5] * d[13] + 2 * d[2] * d[4] * d[14] - 2 * d[1] * d[5] * d[14] -
                 2 * d[1] * d[6] * d[15] + 2 * d[0] * d[7] * d[15] + 2 * d[0] * d[6] * d[16] + 2 * d[1] * d[7] * d[16];
    coeffs[54] = 2 * d[6] * d[7] * d[18] + std::pow(d[2], 2) * d[19] - std::pow(d[5], 2) * d[19] -
                 std::pow(d[6], 2) * d[19] + std::pow(d[7], 2) * d[19] + 2 * d[1] * d[2] * d[20] +
                 2 * d[4] * d[5] * d[20] + 2 * d[2] * d[5] * d[22] + 2 * d[2] * d[4] * d[23] - 2 * d[1] * d[5] * d[23] -
                 2 * d[1] * d[6] * d[24] + 2 * d[0] * d[7] * d[24] + 2 * d[0] * d[6] * d[25] + 2 * d[1] * d[7] * d[25];
    coeffs[55] = 2 * d[2] * d[7] * d[8] - d[1] * std::pow(d[8], 2);
    coeffs[56] = 2 * d[0] * d[1] * d[9] + 2 * d[3] * d[4] * d[9] + std::pow(d[0], 2) * d[10] +
                 3 * std::pow(d[1], 2) * d[10] - std::pow(d[3], 2) * d[10] + std::pow(d[4], 2) * d[10] -
                 2 * d[1] * d[3] * d[12] + 2 * d[0] * d[4] * d[12] + 2 * d[0] * d[3] * d[13] + 2 * d[1] * d[4] * d[13];
    coeffs[57] = 2 * d[0] * d[1] * d[18] + 2 * d[3] * d[4] * d[18] + std::pow(d[0], 2) * d[19] +
                 3 * std::pow(d[1], 2) * d[19] - std::pow(d[3], 2) * d[19] + std::pow(d[4], 2) * d[19] -
                 2 * d[1] * d[3] * d[21] + 2 * d[0] * d[4] * d[21] + 2 * d[0] * d[3] * d[22] + 2 * d[1] * d[4] * d[22];
    coeffs[58] = d[1] * std::pow(d[2], 2) + 2 * d[2] * d[4] * d[5] - d[1] * std::pow(d[5], 2) -
                 d[1] * std::pow(d[6], 2) + 2 * d[0] * d[6] * d[7] + d[1] * std::pow(d[7], 2);
    coeffs[59] = std::pow(d[0], 2) * d[1] + std::pow(d[1], 3) - d[1] * std::pow(d[3], 2) + 2 * d[0] * d[3] * d[4] +
                 d[1] * std::pow(d[4], 2);
    coeffs[60] = d[11] * std::pow(d[17], 2);
    coeffs[61] = std::pow(d[17], 2) * d[20] + 2 * d[11] * d[17] * d[26];
    coeffs[62] = 2 * d[17] * d[20] * d[26] + d[11] * std::pow(d[26], 2);
    coeffs[63] = d[20] * std::pow(d[26], 2);
    coeffs[64] = std::pow(d[11], 3) + d[11] * std::pow(d[14], 2) - d[11] * std::pow(d[15], 2) -
                 d[11] * std::pow(d[16], 2) + 2 * d[9] * d[15] * d[17] + 2 * d[10] * d[16] * d[17];
    coeffs[65] = 2 * d[15] * d[17] * d[18] + 2 * d[16] * d[17] * d[19] + 3 * std::pow(d[11], 2) * d[20] +
                 std::pow(d[14], 2) * d[20] - std::pow(d[15], 2) * d[20] - std::pow(d[16], 2) * d[20] +
                 2 * d[11] * d[14] * d[23] - 2 * d[11] * d[15] * d[24] + 2 * d[9] * d[17] * d[24] -
                 2 * d[11] * d[16] * d[25] + 2 * d[10] * d[17] * d[25] + 2 * d[9] * d[15] * d[26] +
                 2 * d[10] * d[16] * d[26];
    coeffs[66] = 3 * d[11] * std::pow(d[20], 2) + 2 * d[14] * d[20] * d[23] + d[11] * std::pow(d[23], 2) +
                 2 * d[17] * d[18] * d[24] - 2 * d[15] * d[20] * d[24] - d[11] * std::pow(d[24], 2) +
                 2 * d[17] * d[19] * d[25] - 2 * d[16] * d[20] * d[25] - d[11] * std::pow(d[25], 2) +
                 2 * d[15] * d[18] * d[26] + 2 * d[16] * d[19] * d[26] + 2 * d[9] * d[24] * d[26] +
                 2 * d[10] * d[25] * d[26];
    coeffs[67] = std::pow(d[20], 3) + d[20] * std::pow(d[23], 2) - d[20] * std::pow(d[24], 2) -
                 d[20] * std::pow(d[25], 2) + 2 * d[18] * d[24] * d[26] + 2 * d[19] * d[25] * d[26];
    coeffs[68] = 2 * d[8] * d[11] * d[17] + d[2] * std::pow(d[17], 2);
    coeffs[69] = 2 * d[8] * d[17] * d[20] + 2 * d[8] * d[11] * d[26] + 2 * d[2] * d[17] * d[26];
    coeffs[70] = 2 * d[8] * d[20] * d[26] + d[2] * std::pow(d[26], 2);
    coeffs[71] = std::pow(d[9], 2) * d[11] + std::pow(d[10], 2) * d[11] - d[11] * std::pow(d[12], 2) -
                 d[11] * std::pow(d[13], 2) + 2 * d[9] * d[12] * d[14] + 2 * d[10] * d[13] * d[14];
    coeffs[72] = 2 * d[9] * d[11] * d[18] + 2 * d[12] * d[14] * d[18] + 2 * d[10] * d[11] * d[19] +
                 2 * d[13] * d[14] * d[19] + std::pow(d[9], 2) * d[20] + std::pow(d[10], 2) * d[20] -
                 std::pow(d[12], 2) * d[20] - std::pow(d[13], 2) * d[20] - 2 * d[11] * d[12] * d[21] +
                 2 * d[9] * d[14] * d[21] - 2 * d[11] * d[13] * d[22] + 2 * d[10] * d[14] * d[22] +
                 2 * d[9] * d[12] * d[23] + 2 * d[10] * d[13] * d[23];
    coeffs[73] = d[11] * std::pow(d[18], 2) + d[11] * std::pow(d[19], 2) + 2 * d[9] * d[18] * d[20] +
                 2 * d[10] * d[19] * d[20] + 2 * d[14] * d[18] * d[21] - 2 * d[12] * d[20] * d[21] -
                 d[11] * std::pow(d[21], 2) + 2 * d[14] * d[19] * d[22] - 2 * d[13] * d[20] * d[22] -
                 d[11] * std::pow(d[22], 2) + 2 * d[12] * d[18] * d[23] + 2 * d[13] * d[19] * d[23] +
                 2 * d[9] * d[21] * d[23] + 2 * d[10] * d[22] * d[23];
    coeffs[74] = std::pow(d[18], 2) * d[20] + std::pow(d[19], 2) * d[20] - d[20] * std::pow(d[21], 2) -
                 d[20] * std::pow(d[22], 2) + 2 * d[18] * d[21] * d[23] + 2 * d[19] * d[22] * d[23];
    coeffs[75] = 3 * d[2] * std::pow(d[11], 2) + 2 * d[5] * d[11] * d[14] + d[2] * std::pow(d[14], 2) +
                 2 * d[8] * d[9] * d[15] - 2 * d[6] * d[11] * d[15] - d[2] * std::pow(d[15], 2) +
                 2 * d[8] * d[10] * d[16] - 2 * d[7] * d[11] * d[16] - d[2] * std::pow(d[16], 2) +
                 2 * d[6] * d[9] * d[17] + 2 * d[7] * d[10] * d[17] + 2 * d[0] * d[15] * d[17] +
                 2 * d[1] * d[16] * d[17];
    coeffs[76] =
        2 * d[8] * d[15] * d[18] + 2 * d[6] * d[17] * d[18] + 2 * d[8] * d[16] * d[19] + 2 * d[7] * d[17] * d[19] +
        6 * d[2] * d[11] * d[20] + 2 * d[5] * d[14] * d[20] - 2 * d[6] * d[15] * d[20] - 2 * d[7] * d[16] * d[20] +
        2 * d[5] * d[11] * d[23] + 2 * d[2] * d[14] * d[23] + 2 * d[8] * d[9] * d[24] - 2 * d[6] * d[11] * d[24] -
        2 * d[2] * d[15] * d[24] + 2 * d[0] * d[17] * d[24] + 2 * d[8] * d[10] * d[25] - 2 * d[7] * d[11] * d[25] -
        2 * d[2] * d[16] * d[25] + 2 * d[1] * d[17] * d[25] + 2 * d[6] * d[9] * d[26] + 2 * d[7] * d[10] * d[26] +
        2 * d[0] * d[15] * d[26] + 2 * d[1] * d[16] * d[26];
    coeffs[77] = 3 * d[2] * std::pow(d[20], 2) + 2 * d[5] * d[20] * d[23] + d[2] * std::pow(d[23], 2) +
                 2 * d[8] * d[18] * d[24] - 2 * d[6] * d[20] * d[24] - d[2] * std::pow(d[24], 2) +
                 2 * d[8] * d[19] * d[25] - 2 * d[7] * d[20] * d[25] - d[2] * std::pow(d[25], 2) +
                 2 * d[6] * d[18] * d[26] + 2 * d[7] * d[19] * d[26] + 2 * d[0] * d[24] * d[26] +
                 2 * d[1] * d[25] * d[26];
    coeffs[78] = std::pow(d[8], 2) * d[11] + 2 * d[2] * d[8] * d[17];
    coeffs[79] = std::pow(d[8], 2) * d[20] + 2 * d[2] * d[8] * d[26];
    coeffs[80] = d[2] * std::pow(d[9], 2) + d[2] * std::pow(d[10], 2) + 2 * d[0] * d[9] * d[11] +
                 2 * d[1] * d[10] * d[11] + 2 * d[5] * d[9] * d[12] - 2 * d[3] * d[11] * d[12] -
                 d[2] * std::pow(d[12], 2) + 2 * d[5] * d[10] * d[13] - 2 * d[4] * d[11] * d[13] -
                 d[2] * std::pow(d[13], 2) + 2 * d[3] * d[9] * d[14] + 2 * d[4] * d[10] * d[14] +
                 2 * d[0] * d[12] * d[14] + 2 * d[1] * d[13] * d[14];
    coeffs[81] =
        2 * d[2] * d[9] * d[18] + 2 * d[0] * d[11] * d[18] + 2 * d[5] * d[12] * d[18] + 2 * d[3] * d[14] * d[18] +
        2 * d[2] * d[10] * d[19] + 2 * d[1] * d[11] * d[19] + 2 * d[5] * d[13] * d[19] + 2 * d[4] * d[14] * d[19] +
        2 * d[0] * d[9] * d[20] + 2 * d[1] * d[10] * d[20] - 2 * d[3] * d[12] * d[20] - 2 * d[4] * d[13] * d[20] +
        2 * d[5] * d[9] * d[21] - 2 * d[3] * d[11] * d[21] - 2 * d[2] * d[12] * d[21] + 2 * d[0] * d[14] * d[21] +
        2 * d[5] * d[10] * d[22] - 2 * d[4] * d[11] * d[22] - 2 * d[2] * d[13] * d[22] + 2 * d[1] * d[14] * d[22] +
        2 * d[3] * d[9] * d[23] + 2 * d[4] * d[10] * d[23] + 2 * d[0] * d[12] * d[23] + 2 * d[1] * d[13] * d[23];
    coeffs[82] = d[2] * std::pow(d[18], 2) + d[2] * std::pow(d[19], 2) + 2 * d[0] * d[18] * d[20] +
                 2 * d[1] * d[19] * d[20] + 2 * d[5] * d[18] * d[21] - 2 * d[3] * d[20] * d[21] -
                 d[2] * std::pow(d[21], 2) + 2 * d[5] * d[19] * d[22] - 2 * d[4] * d[20] * d[22] -
                 d[2] * std::pow(d[22], 2) + 2 * d[3] * d[18] * d[23] + 2 * d[4] * d[19] * d[23] +
                 2 * d[0] * d[21] * d[23] + 2 * d[1] * d[22] * d[23];
    coeffs[83] = 2 * d[6] * d[8] * d[9] + 2 * d[7] * d[8] * d[10] + 3 * std::pow(d[2], 2) * d[11] +
                 std::pow(d[5], 2) * d[11] - std::pow(d[6], 2) * d[11] - std::pow(d[7], 2) * d[11] +
                 2 * d[2] * d[5] * d[14] - 2 * d[2] * d[6] * d[15] + 2 * d[0] * d[8] * d[15] - 2 * d[2] * d[7] * d[16] +
                 2 * d[1] * d[8] * d[16] + 2 * d[0] * d[6] * d[17] + 2 * d[1] * d[7] * d[17];
    coeffs[84] = 2 * d[6] * d[8] * d[18] + 2 * d[7] * d[8] * d[19] + 3 * std::pow(d[2], 2) * d[20] +
                 std::pow(d[5], 2) * d[20] - std::pow(d[6], 2) * d[20] - std::pow(d[7], 2) * d[20] +
                 2 * d[2] * d[5] * d[23] - 2 * d[2] * d[6] * d[24] + 2 * d[0] * d[8] * d[24] - 2 * d[2] * d[7] * d[25] +
                 2 * d[1] * d[8] * d[25] + 2 * d[0] * d[6] * d[26] + 2 * d[1] * d[7] * d[26];
    coeffs[85] = d[2] * std::pow(d[8], 2);
    coeffs[86] = 2 * d[0] * d[2] * d[9] + 2 * d[3] * d[5] * d[9] + 2 * d[1] * d[2] * d[10] + 2 * d[4] * d[5] * d[10] +
                 std::pow(d[0], 2) * d[11] + std::pow(d[1], 2) * d[11] - std::pow(d[3], 2) * d[11] -
                 std::pow(d[4], 2) * d[11] - 2 * d[2] * d[3] * d[12] + 2 * d[0] * d[5] * d[12] -
                 2 * d[2] * d[4] * d[13] + 2 * d[1] * d[5] * d[13] + 2 * d[0] * d[3] * d[14] + 2 * d[1] * d[4] * d[14];
    coeffs[87] = 2 * d[0] * d[2] * d[18] + 2 * d[3] * d[5] * d[18] + 2 * d[1] * d[2] * d[19] + 2 * d[4] * d[5] * d[19] +
                 std::pow(d[0], 2) * d[20] + std::pow(d[1], 2) * d[20] - std::pow(d[3], 2) * d[20] -
                 std::pow(d[4], 2) * d[20] - 2 * d[2] * d[3] * d[21] + 2 * d[0] * d[5] * d[21] -
                 2 * d[2] * d[4] * d[22] + 2 * d[1] * d[5] * d[22] + 2 * d[0] * d[3] * d[23] + 2 * d[1] * d[4] * d[23];
    coeffs[88] = std::pow(d[2], 3) + d[2] * std::pow(d[5], 2) - d[2] * std::pow(d[6], 2) - d[2] * std::pow(d[7], 2) +
                 2 * d[0] * d[6] * d[8] + 2 * d[1] * d[7] * d[8];
    coeffs[89] = std::pow(d[0], 2) * d[2] + std::pow(d[1], 2) * d[2] - d[2] * std::pow(d[3], 2) -
                 d[2] * std::pow(d[4], 2) + 2 * d[0] * d[3] * d[5] + 2 * d[1] * d[4] * d[5];
    coeffs[90] = 2 * d[14] * d[15] * d[17] - d[12] * std::pow(d[17], 2);
    coeffs[91] = -std::pow(d[17], 2) * d[21] + 2 * d[15] * d[17] * d[23] + 2 * d[14] * d[17] * d[24] +
                 2 * d[14] * d[15] * d[26] - 2 * d[12] * d[17] * d[26];
    coeffs[92] = 2 * d[17] * d[23] * d[24] - 2 * d[17] * d[21] * d[26] + 2 * d[15] * d[23] * d[26] +
                 2 * d[14] * d[24] * d[26] - d[12] * std::pow(d[26], 2);
    coeffs[93] = 2 * d[23] * d[24] * d[26] - d[21] * std::pow(d[26], 2);
    coeffs[94] = -std::pow(d[11], 2) * d[12] + 2 * d[9] * d[11] * d[14] + d[12] * std::pow(d[14], 2) +
                 d[12] * std::pow(d[15], 2) + 2 * d[13] * d[15] * d[16] - d[12] * std::pow(d[16], 2);
    coeffs[95] = 2 * d[11] * d[14] * d[18] - 2 * d[11] * d[12] * d[20] + 2 * d[9] * d[14] * d[20] -
                 std::pow(d[11], 2) * d[21] + std::pow(d[14], 2) * d[21] + std::pow(d[15], 2) * d[21] -
                 std::pow(d[16], 2) * d[21] + 2 * d[15] * d[16] * d[22] + 2 * d[9] * d[11] * d[23] +
                 2 * d[12] * d[14] * d[23] + 2 * d[12] * d[15] * d[24] + 2 * d[13] * d[16] * d[24] +
                 2 * d[13] * d[15] * d[25] - 2 * d[12] * d[16] * d[25];
    coeffs[96] = 2 * d[14] * d[18] * d[20] - d[12] * std::pow(d[20], 2) - 2 * d[11] * d[20] * d[21] +
                 2 * d[11] * d[18] * d[23] + 2 * d[9] * d[20] * d[23] + 2 * d[14] * d[21] * d[23] +
                 d[12] * std::pow(d[23], 2) + 2 * d[15] * d[21] * d[24] + 2 * d[16] * d[22] * d[24] +
                 d[12] * std::pow(d[24], 2) - 2 * d[16] * d[21] * d[25] + 2 * d[15] * d[22] * d[25] +
                 2 * d[13] * d[24] * d[25] - d[12] * std::pow(d[25], 2);
    coeffs[97] = -std::pow(d[20], 2) * d[21] + 2 * d[18] * d[20] * d[23] + d[21] * std::pow(d[23], 2) +
                 d[21] * std::pow(d[24], 2) + 2 * d[22] * d[24] * d[25] - d[21] * std::pow(d[25], 2);
    coeffs[98] = 2 * d[8] * d[14] * d[15] - 2 * d[8] * d[12] * d[17] + 2 * d[6] * d[14] * d[17] +
                 2 * d[5] * d[15] * d[17] - d[3] * std::pow(d[17], 2);
    coeffs[99] = -2 * d[8] * d[17] * d[21] + 2 * d[8] * d[15] * d[23] + 2 * d[6] * d[17] * d[23] +
                 2 * d[8] * d[14] * d[24] + 2 * d[5] * d[17] * d[24] - 2 * d[8] * d[12] * d[26] +
                 2 * d[6] * d[14] * d[26] + 2 * d[5] * d[15] * d[26] - 2 * d[3] * d[17] * d[26];
    coeffs[100] = 2 * d[8] * d[23] * d[24] - 2 * d[8] * d[21] * d[26] + 2 * d[6] * d[23] * d[26] +
                  2 * d[5] * d[24] * d[26] - d[3] * std::pow(d[26], 2);
    coeffs[101] = std::pow(d[9], 2) * d[12] - std::pow(d[10], 2) * d[12] + std::pow(d[12], 3) +
                  2 * d[9] * d[10] * d[13] + d[12] * std::pow(d[13], 2);
    coeffs[102] = 2 * d[9] * d[12] * d[18] + 2 * d[10] * d[13] * d[18] - 2 * d[10] * d[12] * d[19] +
                  2 * d[9] * d[13] * d[19] + std::pow(d[9], 2) * d[21] - std::pow(d[10], 2) * d[21] +
                  3 * std::pow(d[12], 2) * d[21] + std::pow(d[13], 2) * d[21] + 2 * d[9] * d[10] * d[22] +
                  2 * d[12] * d[13] * d[22];
    coeffs[103] = d[12] * std::pow(d[18], 2) + 2 * d[13] * d[18] * d[19] - d[12] * std::pow(d[19], 2) +
                  2 * d[9] * d[18] * d[21] - 2 * d[10] * d[19] * d[21] + 3 * d[12] * std::pow(d[21], 2) +
                  2 * d[10] * d[18] * d[22] + 2 * d[9] * d[19] * d[22] + 2 * d[13] * d[21] * d[22] +
                  d[12] * std::pow(d[22], 2);
    coeffs[104] = std::pow(d[18], 2) * d[21] - std::pow(d[19], 2) * d[21] + std::pow(d[21], 3) +
                  2 * d[18] * d[19] * d[22] + d[21] * std::pow(d[22], 2);
    coeffs[105] = 2 * d[5] * d[9] * d[11] - d[3] * std::pow(d[11], 2) - 2 * d[2] * d[11] * d[12] +
                  2 * d[2] * d[9] * d[14] + 2 * d[0] * d[11] * d[14] + 2 * d[5] * d[12] * d[14] +
                  d[3] * std::pow(d[14], 2) + 2 * d[6] * d[12] * d[15] + 2 * d[7] * d[13] * d[15] +
                  d[3] * std::pow(d[15], 2) - 2 * d[7] * d[12] * d[16] + 2 * d[6] * d[13] * d[16] +
                  2 * d[4] * d[15] * d[16] - d[3] * std::pow(d[16], 2);
    coeffs[106] =
        2 * d[5] * d[11] * d[18] + 2 * d[2] * d[14] * d[18] + 2 * d[5] * d[9] * d[20] - 2 * d[3] * d[11] * d[20] -
        2 * d[2] * d[12] * d[20] + 2 * d[0] * d[14] * d[20] - 2 * d[2] * d[11] * d[21] + 2 * d[5] * d[14] * d[21] +
        2 * d[6] * d[15] * d[21] - 2 * d[7] * d[16] * d[21] + 2 * d[7] * d[15] * d[22] + 2 * d[6] * d[16] * d[22] +
        2 * d[2] * d[9] * d[23] + 2 * d[0] * d[11] * d[23] + 2 * d[5] * d[12] * d[23] + 2 * d[3] * d[14] * d[23] +
        2 * d[6] * d[12] * d[24] + 2 * d[7] * d[13] * d[24] + 2 * d[3] * d[15] * d[24] + 2 * d[4] * d[16] * d[24] -
        2 * d[7] * d[12] * d[25] + 2 * d[6] * d[13] * d[25] + 2 * d[4] * d[15] * d[25] - 2 * d[3] * d[16] * d[25];
    coeffs[107] = 2 * d[5] * d[18] * d[20] - d[3] * std::pow(d[20], 2) - 2 * d[2] * d[20] * d[21] +
                  2 * d[2] * d[18] * d[23] + 2 * d[0] * d[20] * d[23] + 2 * d[5] * d[21] * d[23] +
                  d[3] * std::pow(d[23], 2) + 2 * d[6] * d[21] * d[24] + 2 * d[7] * d[22] * d[24] +
                  d[3] * std::pow(d[24], 2) - 2 * d[7] * d[21] * d[25] + 2 * d[6] * d[22] * d[25] +
                  2 * d[4] * d[24] * d[25] - d[3] * std::pow(d[25], 2);
    coeffs[108] = -std::pow(d[8], 2) * d[12] + 2 * d[6] * d[8] * d[14] + 2 * d[5] * d[8] * d[15] +
                  2 * d[5] * d[6] * d[17] - 2 * d[3] * d[8] * d[17];
    coeffs[109] = -std::pow(d[8], 2) * d[21] + 2 * d[6] * d[8] * d[23] + 2 * d[5] * d[8] * d[24] +
                  2 * d[5] * d[6] * d[26] - 2 * d[3] * d[8] * d[26];
    coeffs[110] = d[3] * std::pow(d[9], 2) + 2 * d[4] * d[9] * d[10] - d[3] * std::pow(d[10], 2) +
                  2 * d[0] * d[9] * d[12] - 2 * d[1] * d[10] * d[12] + 3 * d[3] * std::pow(d[12], 2) +
                  2 * d[1] * d[9] * d[13] + 2 * d[0] * d[10] * d[13] + 2 * d[4] * d[12] * d[13] +
                  d[3] * std::pow(d[13], 2);
    coeffs[111] =
        2 * d[3] * d[9] * d[18] + 2 * d[4] * d[10] * d[18] + 2 * d[0] * d[12] * d[18] + 2 * d[1] * d[13] * d[18] +
        2 * d[4] * d[9] * d[19] - 2 * d[3] * d[10] * d[19] - 2 * d[1] * d[12] * d[19] + 2 * d[0] * d[13] * d[19] +
        2 * d[0] * d[9] * d[21] - 2 * d[1] * d[10] * d[21] + 6 * d[3] * d[12] * d[21] + 2 * d[4] * d[13] * d[21] +
        2 * d[1] * d[9] * d[22] + 2 * d[0] * d[10] * d[22] + 2 * d[4] * d[12] * d[22] + 2 * d[3] * d[13] * d[22];
    coeffs[112] = d[3] * std::pow(d[18], 2) + 2 * d[4] * d[18] * d[19] - d[3] * std::pow(d[19], 2) +
                  2 * d[0] * d[18] * d[21] - 2 * d[1] * d[19] * d[21] + 3 * d[3] * std::pow(d[21], 2) +
                  2 * d[1] * d[18] * d[22] + 2 * d[0] * d[19] * d[22] + 2 * d[4] * d[21] * d[22] +
                  d[3] * std::pow(d[22], 2);
    coeffs[113] = 2 * d[2] * d[5] * d[9] - 2 * d[2] * d[3] * d[11] + 2 * d[0] * d[5] * d[11] -
                  std::pow(d[2], 2) * d[12] + std::pow(d[5], 2) * d[12] + std::pow(d[6], 2) * d[12] -
                  std::pow(d[7], 2) * d[12] + 2 * d[6] * d[7] * d[13] + 2 * d[0] * d[2] * d[14] +
                  2 * d[3] * d[5] * d[14] + 2 * d[3] * d[6] * d[15] + 2 * d[4] * d[7] * d[15] +
                  2 * d[4] * d[6] * d[16] - 2 * d[3] * d[7] * d[16];
    coeffs[114] = 2 * d[2] * d[5] * d[18] - 2 * d[2] * d[3] * d[20] + 2 * d[0] * d[5] * d[20] -
                  std::pow(d[2], 2) * d[21] + std::pow(d[5], 2) * d[21] + std::pow(d[6], 2) * d[21] -
                  std::pow(d[7], 2) * d[21] + 2 * d[6] * d[7] * d[22] + 2 * d[0] * d[2] * d[23] +
                  2 * d[3] * d[5] * d[23] + 2 * d[3] * d[6] * d[24] + 2 * d[4] * d[7] * d[24] +
                  2 * d[4] * d[6] * d[25] - 2 * d[3] * d[7] * d[25];
    coeffs[115] = 2 * d[5] * d[6] * d[8] - d[3] * std::pow(d[8], 2);
    coeffs[116] = 2 * d[0] * d[3] * d[9] + 2 * d[1] * d[4] * d[9] - 2 * d[1] * d[3] * d[10] + 2 * d[0] * d[4] * d[10] +
                  std::pow(d[0], 2) * d[12] - std::pow(d[1], 2) * d[12] + 3 * std::pow(d[3], 2) * d[12] +
                  std::pow(d[4], 2) * d[12] + 2 * d[0] * d[1] * d[13] + 2 * d[3] * d[4] * d[13];
    coeffs[117] = 2 * d[0] * d[3] * d[18] + 2 * d[1] * d[4] * d[18] - 2 * d[1] * d[3] * d[19] +
                  2 * d[0] * d[4] * d[19] + std::pow(d[0], 2) * d[21] - std::pow(d[1], 2) * d[21] +
                  3 * std::pow(d[3], 2) * d[21] + std::pow(d[4], 2) * d[21] + 2 * d[0] * d[1] * d[22] +
                  2 * d[3] * d[4] * d[22];
    coeffs[118] = -std::pow(d[2], 2) * d[3] + 2 * d[0] * d[2] * d[5] + d[3] * std::pow(d[5], 2) +
                  d[3] * std::pow(d[6], 2) + 2 * d[4] * d[6] * d[7] - d[3] * std::pow(d[7], 2);
    coeffs[119] = std::pow(d[0], 2) * d[3] - std::pow(d[1], 2) * d[3] + std::pow(d[3], 3) + 2 * d[0] * d[1] * d[4] +
                  d[3] * std::pow(d[4], 2);
    coeffs[120] = 2 * d[14] * d[16] * d[17] - d[13] * std::pow(d[17], 2);
    coeffs[121] = -std::pow(d[17], 2) * d[22] + 2 * d[16] * d[17] * d[23] + 2 * d[14] * d[17] * d[25] +
                  2 * d[14] * d[16] * d[26] - 2 * d[13] * d[17] * d[26];
    coeffs[122] = 2 * d[17] * d[23] * d[25] - 2 * d[17] * d[22] * d[26] + 2 * d[16] * d[23] * d[26] +
                  2 * d[14] * d[25] * d[26] - d[13] * std::pow(d[26], 2);
    coeffs[123] = 2 * d[23] * d[25] * d[26] - d[22] * std::pow(d[26], 2);
    coeffs[124] = -std::pow(d[11], 2) * d[13] + 2 * d[10] * d[11] * d[14] + d[13] * std::pow(d[14], 2) -
                  d[13] * std::pow(d[15], 2) + 2 * d[12] * d[15] * d[16] + d[13] * std::pow(d[16], 2);
    coeffs[125] = 2 * d[11] * d[14] * d[19] - 2 * d[11] * d[13] * d[20] + 2 * d[10] * d[14] * d[20] +
                  2 * d[15] * d[16] * d[21] - std::pow(d[11], 2) * d[22] + std::pow(d[14], 2) * d[22] -
                  std::pow(d[15], 2) * d[22] + std::pow(d[16], 2) * d[22] + 2 * d[10] * d[11] * d[23] +
                  2 * d[13] * d[14] * d[23] - 2 * d[13] * d[15] * d[24] + 2 * d[12] * d[16] * d[24] +
                  2 * d[12] * d[15] * d[25] + 2 * d[13] * d[16] * d[25];
    coeffs[126] = 2 * d[14] * d[19] * d[20] - d[13] * std::pow(d[20], 2) - 2 * d[11] * d[20] * d[22] +
                  2 * d[11] * d[19] * d[23] + 2 * d[10] * d[20] * d[23] + 2 * d[14] * d[22] * d[23] +
                  d[13] * std::pow(d[23], 2) + 2 * d[16] * d[21] * d[24] - 2 * d[15] * d[22] * d[24] -
                  d[13] * std::pow(d[24], 2) + 2 * d[15] * d[21] * d[25] + 2 * d[16] * d[22] * d[25] +
                  2 * d[12] * d[24] * d[25] + d[13] * std::pow(d[25], 2);
    coeffs[127] = -std::pow(d[20], 2) * d[22] + 2 * d[19] * d[20] * d[23] + d[22] * std::pow(d[23], 2) -
                  d[22] * std::pow(d[24], 2) + 2 * d[21] * d[24] * d[25] + d[22] * std::pow(d[25], 2);
    coeffs[128] = 2 * d[8] * d[14] * d[16] - 2 * d[8] * d[13] * d[17] + 2 * d[7] * d[14] * d[17] +
                  2 * d[5] * d[16] * d[17] - d[4] * std::pow(d[17], 2);
    coeffs[129] = -2 * d[8] * d[17] * d[22] + 2 * d[8] * d[16] * d[23] + 2 * d[7] * d[17] * d[23] +
                  2 * d[8] * d[14] * d[25] + 2 * d[5] * d[17] * d[25] - 2 * d[8] * d[13] * d[26] +
                  2 * d[7] * d[14] * d[26] + 2 * d[5] * d[16] * d[26] - 2 * d[4] * d[17] * d[26];
    coeffs[130] = 2 * d[8] * d[23] * d[25] - 2 * d[8] * d[22] * d[26] + 2 * d[7] * d[23] * d[26] +
                  2 * d[5] * d[25] * d[26] - d[4] * std::pow(d[26], 2);
    coeffs[131] = 2 * d[9] * d[10] * d[12] - std::pow(d[9], 2) * d[13] + std::pow(d[10], 2) * d[13] +
                  std::pow(d[12], 2) * d[13] + std::pow(d[13], 3);
    coeffs[132] = 2 * d[10] * d[12] * d[18] - 2 * d[9] * d[13] * d[18] + 2 * d[9] * d[12] * d[19] +
                  2 * d[10] * d[13] * d[19] + 2 * d[9] * d[10] * d[21] + 2 * d[12] * d[13] * d[21] -
                  std::pow(d[9], 2) * d[22] + std::pow(d[10], 2) * d[22] + std::pow(d[12], 2) * d[22] +
                  3 * std::pow(d[13], 2) * d[22];
    coeffs[133] = -d[13] * std::pow(d[18], 2) + 2 * d[12] * d[18] * d[19] + d[13] * std::pow(d[19], 2) +
                  2 * d[10] * d[18] * d[21] + 2 * d[9] * d[19] * d[21] + d[13] * std::pow(d[21], 2) -
                  2 * d[9] * d[18] * d[22] + 2 * d[10] * d[19] * d[22] + 2 * d[12] * d[21] * d[22] +
                  3 * d[13] * std::pow(d[22], 2);
    coeffs[134] = 2 * d[18] * d[19] * d[21] - std::pow(d[18], 2) * d[22] + std::pow(d[19], 2) * d[22] +
                  std::pow(d[21], 2) * d[22] + std::pow(d[22], 3);
    coeffs[135] = 2 * d[5] * d[10] * d[11] - d[4] * std::pow(d[11], 2) - 2 * d[2] * d[11] * d[13] +
                  2 * d[2] * d[10] * d[14] + 2 * d[1] * d[11] * d[14] + 2 * d[5] * d[13] * d[14] +
                  d[4] * std::pow(d[14], 2) + 2 * d[7] * d[12] * d[15] - 2 * d[6] * d[13] * d[15] -
                  d[4] * std::pow(d[15], 2) + 2 * d[6] * d[12] * d[16] + 2 * d[7] * d[13] * d[16] +
                  2 * d[3] * d[15] * d[16] + d[4] * std::pow(d[16], 2);
    coeffs[136] =
        2 * d[5] * d[11] * d[19] + 2 * d[2] * d[14] * d[19] + 2 * d[5] * d[10] * d[20] - 2 * d[4] * d[11] * d[20] -
        2 * d[2] * d[13] * d[20] + 2 * d[1] * d[14] * d[20] + 2 * d[7] * d[15] * d[21] + 2 * d[6] * d[16] * d[21] -
        2 * d[2] * d[11] * d[22] + 2 * d[5] * d[14] * d[22] - 2 * d[6] * d[15] * d[22] + 2 * d[7] * d[16] * d[22] +
        2 * d[2] * d[10] * d[23] + 2 * d[1] * d[11] * d[23] + 2 * d[5] * d[13] * d[23] + 2 * d[4] * d[14] * d[23] +
        2 * d[7] * d[12] * d[24] - 2 * d[6] * d[13] * d[24] - 2 * d[4] * d[15] * d[24] + 2 * d[3] * d[16] * d[24] +
        2 * d[6] * d[12] * d[25] + 2 * d[7] * d[13] * d[25] + 2 * d[3] * d[15] * d[25] + 2 * d[4] * d[16] * d[25];
    coeffs[137] = 2 * d[5] * d[19] * d[20] - d[4] * std::pow(d[20], 2) - 2 * d[2] * d[20] * d[22] +
                  2 * d[2] * d[19] * d[23] + 2 * d[1] * d[20] * d[23] + 2 * d[5] * d[22] * d[23] +
                  d[4] * std::pow(d[23], 2) + 2 * d[7] * d[21] * d[24] - 2 * d[6] * d[22] * d[24] -
                  d[4] * std::pow(d[24], 2) + 2 * d[6] * d[21] * d[25] + 2 * d[7] * d[22] * d[25] +
                  2 * d[3] * d[24] * d[25] + d[4] * std::pow(d[25], 2);
    coeffs[138] = -std::pow(d[8], 2) * d[13] + 2 * d[7] * d[8] * d[14] + 2 * d[5] * d[8] * d[16] +
                  2 * d[5] * d[7] * d[17] - 2 * d[4] * d[8] * d[17];
    coeffs[139] = -std::pow(d[8], 2) * d[22] + 2 * d[7] * d[8] * d[23] + 2 * d[5] * d[8] * d[25] +
                  2 * d[5] * d[7] * d[26] - 2 * d[4] * d[8] * d[26];
    coeffs[140] = -d[4] * std::pow(d[9], 2) + 2 * d[3] * d[9] * d[10] + d[4] * std::pow(d[10], 2) +
                  2 * d[1] * d[9] * d[12] + 2 * d[0] * d[10] * d[12] + d[4] * std::pow(d[12], 2) -
                  2 * d[0] * d[9] * d[13] + 2 * d[1] * d[10] * d[13] + 2 * d[3] * d[12] * d[13] +
                  3 * d[4] * std::pow(d[13], 2);
    coeffs[141] =
        -2 * d[4] * d[9] * d[18] + 2 * d[3] * d[10] * d[18] + 2 * d[1] * d[12] * d[18] - 2 * d[0] * d[13] * d[18] +
        2 * d[3] * d[9] * d[19] + 2 * d[4] * d[10] * d[19] + 2 * d[0] * d[12] * d[19] + 2 * d[1] * d[13] * d[19] +
        2 * d[1] * d[9] * d[21] + 2 * d[0] * d[10] * d[21] + 2 * d[4] * d[12] * d[21] + 2 * d[3] * d[13] * d[21] -
        2 * d[0] * d[9] * d[22] + 2 * d[1] * d[10] * d[22] + 2 * d[3] * d[12] * d[22] + 6 * d[4] * d[13] * d[22];
    coeffs[142] = -d[4] * std::pow(d[18], 2) + 2 * d[3] * d[18] * d[19] + d[4] * std::pow(d[19], 2) +
                  2 * d[1] * d[18] * d[21] + 2 * d[0] * d[19] * d[21] + d[4] * std::pow(d[21], 2) -
                  2 * d[0] * d[18] * d[22] + 2 * d[1] * d[19] * d[22] + 2 * d[3] * d[21] * d[22] +
                  3 * d[4] * std::pow(d[22], 2);
    coeffs[143] = 2 * d[2] * d[5] * d[10] - 2 * d[2] * d[4] * d[11] + 2 * d[1] * d[5] * d[11] +
                  2 * d[6] * d[7] * d[12] - std::pow(d[2], 2) * d[13] + std::pow(d[5], 2) * d[13] -
                  std::pow(d[6], 2) * d[13] + std::pow(d[7], 2) * d[13] + 2 * d[1] * d[2] * d[14] +
                  2 * d[4] * d[5] * d[14] - 2 * d[4] * d[6] * d[15] + 2 * d[3] * d[7] * d[15] +
                  2 * d[3] * d[6] * d[16] + 2 * d[4] * d[7] * d[16];
    coeffs[144] = 2 * d[2] * d[5] * d[19] - 2 * d[2] * d[4] * d[20] + 2 * d[1] * d[5] * d[20] +
                  2 * d[6] * d[7] * d[21] - std::pow(d[2], 2) * d[22] + std::pow(d[5], 2) * d[22] -
                  std::pow(d[6], 2) * d[22] + std::pow(d[7], 2) * d[22] + 2 * d[1] * d[2] * d[23] +
                  2 * d[4] * d[5] * d[23] - 2 * d[4] * d[6] * d[24] + 2 * d[3] * d[7] * d[24] +
                  2 * d[3] * d[6] * d[25] + 2 * d[4] * d[7] * d[25];
    coeffs[145] = 2 * d[5] * d[7] * d[8] - d[4] * std::pow(d[8], 2);
    coeffs[146] = 2 * d[1] * d[3] * d[9] - 2 * d[0] * d[4] * d[9] + 2 * d[0] * d[3] * d[10] + 2 * d[1] * d[4] * d[10] +
                  2 * d[0] * d[1] * d[12] + 2 * d[3] * d[4] * d[12] - std::pow(d[0], 2) * d[13] +
                  std::pow(d[1], 2) * d[13] + std::pow(d[3], 2) * d[13] + 3 * std::pow(d[4], 2) * d[13];
    coeffs[147] = 2 * d[1] * d[3] * d[18] - 2 * d[0] * d[4] * d[18] + 2 * d[0] * d[3] * d[19] +
                  2 * d[1] * d[4] * d[19] + 2 * d[0] * d[1] * d[21] + 2 * d[3] * d[4] * d[21] -
                  std::pow(d[0], 2) * d[22] + std::pow(d[1], 2) * d[22] + std::pow(d[3], 2) * d[22] +
                  3 * std::pow(d[4], 2) * d[22];
    coeffs[148] = -std::pow(d[2], 2) * d[4] + 2 * d[1] * d[2] * d[5] + d[4] * std::pow(d[5], 2) -
                  d[4] * std::pow(d[6], 2) + 2 * d[3] * d[6] * d[7] + d[4] * std::pow(d[7], 2);
    coeffs[149] = 2 * d[0] * d[1] * d[3] - std::pow(d[0], 2) * d[4] + std::pow(d[1], 2) * d[4] +
                  std::pow(d[3], 2) * d[4] + std::pow(d[4], 3);
    coeffs[150] = d[14] * std::pow(d[17], 2);
    coeffs[151] = std::pow(d[17], 2) * d[23] + 2 * d[14] * d[17] * d[26];
    coeffs[152] = 2 * d[17] * d[23] * d[26] + d[14] * std::pow(d[26], 2);
    coeffs[153] = d[23] * std::pow(d[26], 2);
    coeffs[154] = std::pow(d[11], 2) * d[14] + std::pow(d[14], 3) - d[14] * std::pow(d[15], 2) -
                  d[14] * std::pow(d[16], 2) + 2 * d[12] * d[15] * d[17] + 2 * d[13] * d[16] * d[17];
    coeffs[155] = 2 * d[11] * d[14] * d[20] + 2 * d[15] * d[17] * d[21] + 2 * d[16] * d[17] * d[22] +
                  std::pow(d[11], 2) * d[23] + 3 * std::pow(d[14], 2) * d[23] - std::pow(d[15], 2) * d[23] -
                  std::pow(d[16], 2) * d[23] - 2 * d[14] * d[15] * d[24] + 2 * d[12] * d[17] * d[24] -
                  2 * d[14] * d[16] * d[25] + 2 * d[13] * d[17] * d[25] + 2 * d[12] * d[15] * d[26] +
                  2 * d[13] * d[16] * d[26];
    coeffs[156] = d[14] * std::pow(d[20], 2) + 2 * d[11] * d[20] * d[23] + 3 * d[14] * std::pow(d[23], 2) +
                  2 * d[17] * d[21] * d[24] - 2 * d[15] * d[23] * d[24] - d[14] * std::pow(d[24], 2) +
                  2 * d[17] * d[22] * d[25] - 2 * d[16] * d[23] * d[25] - d[14] * std::pow(d[25], 2) +
                  2 * d[15] * d[21] * d[26] + 2 * d[16] * d[22] * d[26] + 2 * d[12] * d[24] * d[26] +
                  2 * d[13] * d[25] * d[26];
    coeffs[157] = std::pow(d[20], 2) * d[23] + std::pow(d[23], 3) - d[23] * std::pow(d[24], 2) -
                  d[23] * std::pow(d[25], 2) + 2 * d[21] * d[24] * d[26] + 2 * d[22] * d[25] * d[26];
    coeffs[158] = 2 * d[8] * d[14] * d[17] + d[5] * std::pow(d[17], 2);
    coeffs[159] = 2 * d[8] * d[17] * d[23] + 2 * d[8] * d[14] * d[26] + 2 * d[5] * d[17] * d[26];
    coeffs[160] = 2 * d[8] * d[23] * d[26] + d[5] * std::pow(d[26], 2);
    coeffs[161] = 2 * d[9] * d[11] * d[12] + 2 * d[10] * d[11] * d[13] - std::pow(d[9], 2) * d[14] -
                  std::pow(d[10], 2) * d[14] + std::pow(d[12], 2) * d[14] + std::pow(d[13], 2) * d[14];
    coeffs[162] = 2 * d[11] * d[12] * d[18] - 2 * d[9] * d[14] * d[18] + 2 * d[11] * d[13] * d[19] -
                  2 * d[10] * d[14] * d[19] + 2 * d[9] * d[12] * d[20] + 2 * d[10] * d[13] * d[20] +
                  2 * d[9] * d[11] * d[21] + 2 * d[12] * d[14] * d[21] + 2 * d[10] * d[11] * d[22] +
                  2 * d[13] * d[14] * d[22] - std::pow(d[9], 2) * d[23] - std::pow(d[10], 2) * d[23] +
                  std::pow(d[12], 2) * d[23] + std::pow(d[13], 2) * d[23];
    coeffs[163] = -d[14] * std::pow(d[18], 2) - d[14] * std::pow(d[19], 2) + 2 * d[12] * d[18] * d[20] +
                  2 * d[13] * d[19] * d[20] + 2 * d[11] * d[18] * d[21] + 2 * d[9] * d[20] * d[21] +
                  d[14] * std::pow(d[21], 2) + 2 * d[11] * d[19] * d[22] + 2 * d[10] * d[20] * d[22] +
                  d[14] * std::pow(d[22], 2) - 2 * d[9] * d[18] * d[23] - 2 * d[10] * d[19] * d[23] +
                  2 * d[12] * d[21] * d[23] + 2 * d[13] * d[22] * d[23];
    coeffs[164] = 2 * d[18] * d[20] * d[21] + 2 * d[19] * d[20] * d[22] - std::pow(d[18], 2) * d[23] -
                  std::pow(d[19], 2) * d[23] + std::pow(d[21], 2) * d[23] + std::pow(d[22], 2) * d[23];
    coeffs[165] = d[5] * std::pow(d[11], 2) + 2 * d[2] * d[11] * d[14] + 3 * d[5] * std::pow(d[14], 2) +
                  2 * d[8] * d[12] * d[15] - 2 * d[6] * d[14] * d[15] - d[5] * std::pow(d[15], 2) +
                  2 * d[8] * d[13] * d[16] - 2 * d[7] * d[14] * d[16] - d[5] * std::pow(d[16], 2) +
                  2 * d[6] * d[12] * d[17] + 2 * d[7] * d[13] * d[17] + 2 * d[3] * d[15] * d[17] +
                  2 * d[4] * d[16] * d[17];
    coeffs[166] =
        2 * d[5] * d[11] * d[20] + 2 * d[2] * d[14] * d[20] + 2 * d[8] * d[15] * d[21] + 2 * d[6] * d[17] * d[21] +
        2 * d[8] * d[16] * d[22] + 2 * d[7] * d[17] * d[22] + 2 * d[2] * d[11] * d[23] + 6 * d[5] * d[14] * d[23] -
        2 * d[6] * d[15] * d[23] - 2 * d[7] * d[16] * d[23] + 2 * d[8] * d[12] * d[24] - 2 * d[6] * d[14] * d[24] -
        2 * d[5] * d[15] * d[24] + 2 * d[3] * d[17] * d[24] + 2 * d[8] * d[13] * d[25] - 2 * d[7] * d[14] * d[25] -
        2 * d[5] * d[16] * d[25] + 2 * d[4] * d[17] * d[25] + 2 * d[6] * d[12] * d[26] + 2 * d[7] * d[13] * d[26] +
        2 * d[3] * d[15] * d[26] + 2 * d[4] * d[16] * d[26];
    coeffs[167] = d[5] * std::pow(d[20], 2) + 2 * d[2] * d[20] * d[23] + 3 * d[5] * std::pow(d[23], 2) +
                  2 * d[8] * d[21] * d[24] - 2 * d[6] * d[23] * d[24] - d[5] * std::pow(d[24], 2) +
                  2 * d[8] * d[22] * d[25] - 2 * d[7] * d[23] * d[25] - d[5] * std::pow(d[25], 2) +
                  2 * d[6] * d[21] * d[26] + 2 * d[7] * d[22] * d[26] + 2 * d[3] * d[24] * d[26] +
                  2 * d[4] * d[25] * d[26];
    coeffs[168] = std::pow(d[8], 2) * d[14] + 2 * d[5] * d[8] * d[17];
    coeffs[169] = std::pow(d[8], 2) * d[23] + 2 * d[5] * d[8] * d[26];
    coeffs[170] = -d[5] * std::pow(d[9], 2) - d[5] * std::pow(d[10], 2) + 2 * d[3] * d[9] * d[11] +
                  2 * d[4] * d[10] * d[11] + 2 * d[2] * d[9] * d[12] + 2 * d[0] * d[11] * d[12] +
                  d[5] * std::pow(d[12], 2) + 2 * d[2] * d[10] * d[13] + 2 * d[1] * d[11] * d[13] +
                  d[5] * std::pow(d[13], 2) - 2 * d[0] * d[9] * d[14] - 2 * d[1] * d[10] * d[14] +
                  2 * d[3] * d[12] * d[14] + 2 * d[4] * d[13] * d[14];
    coeffs[171] =
        -2 * d[5] * d[9] * d[18] + 2 * d[3] * d[11] * d[18] + 2 * d[2] * d[12] * d[18] - 2 * d[0] * d[14] * d[18] -
        2 * d[5] * d[10] * d[19] + 2 * d[4] * d[11] * d[19] + 2 * d[2] * d[13] * d[19] - 2 * d[1] * d[14] * d[19] +
        2 * d[3] * d[9] * d[20] + 2 * d[4] * d[10] * d[20] + 2 * d[0] * d[12] * d[20] + 2 * d[1] * d[13] * d[20] +
        2 * d[2] * d[9] * d[21] + 2 * d[0] * d[11] * d[21] + 2 * d[5] * d[12] * d[21] + 2 * d[3] * d[14] * d[21] +
        2 * d[2] * d[10] * d[22] + 2 * d[1] * d[11] * d[22] + 2 * d[5] * d[13] * d[22] + 2 * d[4] * d[14] * d[22] -
        2 * d[0] * d[9] * d[23] - 2 * d[1] * d[10] * d[23] + 2 * d[3] * d[12] * d[23] + 2 * d[4] * d[13] * d[23];
    coeffs[172] = -d[5] * std::pow(d[18], 2) - d[5] * std::pow(d[19], 2) + 2 * d[3] * d[18] * d[20] +
                  2 * d[4] * d[19] * d[20] + 2 * d[2] * d[18] * d[21] + 2 * d[0] * d[20] * d[21] +
                  d[5] * std::pow(d[21], 2) + 2 * d[2] * d[19] * d[22] + 2 * d[1] * d[20] * d[22] +
                  d[5] * std::pow(d[22], 2) - 2 * d[0] * d[18] * d[23] - 2 * d[1] * d[19] * d[23] +
                  2 * d[3] * d[21] * d[23] + 2 * d[4] * d[22] * d[23];
    coeffs[173] = 2 * d[2] * d[5] * d[11] + 2 * d[6] * d[8] * d[12] + 2 * d[7] * d[8] * d[13] +
                  std::pow(d[2], 2) * d[14] + 3 * std::pow(d[5], 2) * d[14] - std::pow(d[6], 2) * d[14] -
                  std::pow(d[7], 2) * d[14] - 2 * d[5] * d[6] * d[15] + 2 * d[3] * d[8] * d[15] -
                  2 * d[5] * d[7] * d[16] + 2 * d[4] * d[8] * d[16] + 2 * d[3] * d[6] * d[17] + 2 * d[4] * d[7] * d[17];
    coeffs[174] = 2 * d[2] * d[5] * d[20] + 2 * d[6] * d[8] * d[21] + 2 * d[7] * d[8] * d[22] +
                  std::pow(d[2], 2) * d[23] + 3 * std::pow(d[5], 2) * d[23] - std::pow(d[6], 2) * d[23] -
                  std::pow(d[7], 2) * d[23] - 2 * d[5] * d[6] * d[24] + 2 * d[3] * d[8] * d[24] -
                  2 * d[5] * d[7] * d[25] + 2 * d[4] * d[8] * d[25] + 2 * d[3] * d[6] * d[26] + 2 * d[4] * d[7] * d[26];
    coeffs[175] = d[5] * std::pow(d[8], 2);
    coeffs[176] = 2 * d[2] * d[3] * d[9] - 2 * d[0] * d[5] * d[9] + 2 * d[2] * d[4] * d[10] - 2 * d[1] * d[5] * d[10] +
                  2 * d[0] * d[3] * d[11] + 2 * d[1] * d[4] * d[11] + 2 * d[0] * d[2] * d[12] +
                  2 * d[3] * d[5] * d[12] + 2 * d[1] * d[2] * d[13] + 2 * d[4] * d[5] * d[13] -
                  std::pow(d[0], 2) * d[14] - std::pow(d[1], 2) * d[14] + std::pow(d[3], 2) * d[14] +
                  std::pow(d[4], 2) * d[14];
    coeffs[177] = 2 * d[2] * d[3] * d[18] - 2 * d[0] * d[5] * d[18] + 2 * d[2] * d[4] * d[19] -
                  2 * d[1] * d[5] * d[19] + 2 * d[0] * d[3] * d[20] + 2 * d[1] * d[4] * d[20] +
                  2 * d[0] * d[2] * d[21] + 2 * d[3] * d[5] * d[21] + 2 * d[1] * d[2] * d[22] +
                  2 * d[4] * d[5] * d[22] - std::pow(d[0], 2) * d[23] - std::pow(d[1], 2) * d[23] +
                  std::pow(d[3], 2) * d[23] + std::pow(d[4], 2) * d[23];
    coeffs[178] = std::pow(d[2], 2) * d[5] + std::pow(d[5], 3) - d[5] * std::pow(d[6], 2) - d[5] * std::pow(d[7], 2) +
                  2 * d[3] * d[6] * d[8] + 2 * d[4] * d[7] * d[8];
    coeffs[179] = 2 * d[0] * d[2] * d[3] + 2 * d[1] * d[2] * d[4] - std::pow(d[0], 2) * d[5] -
                  std::pow(d[1], 2) * d[5] + std::pow(d[3], 2) * d[5] + std::pow(d[4], 2) * d[5];
    coeffs[180] = d[15] * std::pow(d[17], 2);
    coeffs[181] = std::pow(d[17], 2) * d[24] + 2 * d[15] * d[17] * d[26];
    coeffs[182] = 2 * d[17] * d[24] * d[26] + d[15] * std::pow(d[26], 2);
    coeffs[183] = d[24] * std::pow(d[26], 2);
    coeffs[184] = -std::pow(d[11], 2) * d[15] - std::pow(d[14], 2) * d[15] + std::pow(d[15], 3) +
                  d[15] * std::pow(d[16], 2) + 2 * d[9] * d[11] * d[17] + 2 * d[12] * d[14] * d[17];
    coeffs[185] = 2 * d[11] * d[17] * d[18] - 2 * d[11] * d[15] * d[20] + 2 * d[9] * d[17] * d[20] +
                  2 * d[14] * d[17] * d[21] - 2 * d[14] * d[15] * d[23] + 2 * d[12] * d[17] * d[23] -
                  std::pow(d[11], 2) * d[24] - std::pow(d[14], 2) * d[24] + 3 * std::pow(d[15], 2) * d[24] +
                  std::pow(d[16], 2) * d[24] + 2 * d[15] * d[16] * d[25] + 2 * d[9] * d[11] * d[26] +
                  2 * d[12] * d[14] * d[26];
    coeffs[186] = 2 * d[17] * d[18] * d[20] - d[15] * std::pow(d[20], 2) + 2 * d[17] * d[21] * d[23] -
                  d[15] * std::pow(d[23], 2) - 2 * d[11] * d[20] * d[24] - 2 * d[14] * d[23] * d[24] +
                  3 * d[15] * std::pow(d[24], 2) + 2 * d[16] * d[24] * d[25] + d[15] * std::pow(d[25], 2) +
                  2 * d[11] * d[18] * d[26] + 2 * d[9] * d[20] * d[26] + 2 * d[14] * d[21] * d[26] +
                  2 * d[12] * d[23] * d[26];
    coeffs[187] = -std::pow(d[20], 2) * d[24] - std::pow(d[23], 2) * d[24] + std::pow(d[24], 3) +
                  d[24] * std::pow(d[25], 2) + 2 * d[18] * d[20] * d[26] + 2 * d[21] * d[23] * d[26];
    coeffs[188] = 2 * d[8] * d[15] * d[17] + d[6] * std::pow(d[17], 2);
    coeffs[189] = 2 * d[8] * d[17] * d[24] + 2 * d[8] * d[15] * d[26] + 2 * d[6] * d[17] * d[26];
    coeffs[190] = 2 * d[8] * d[24] * d[26] + d[6] * std::pow(d[26], 2);
    coeffs[191] = std::pow(d[9], 2) * d[15] - std::pow(d[10], 2) * d[15] + std::pow(d[12], 2) * d[15] -
                  std::pow(d[13], 2) * d[15] + 2 * d[9] * d[10] * d[16] + 2 * d[12] * d[13] * d[16];
    coeffs[192] = 2 * d[9] * d[15] * d[18] + 2 * d[10] * d[16] * d[18] - 2 * d[10] * d[15] * d[19] +
                  2 * d[9] * d[16] * d[19] + 2 * d[12] * d[15] * d[21] + 2 * d[13] * d[16] * d[21] -
                  2 * d[13] * d[15] * d[22] + 2 * d[12] * d[16] * d[22] + std::pow(d[9], 2) * d[24] -
                  std::pow(d[10], 2) * d[24] + std::pow(d[12], 2) * d[24] - std::pow(d[13], 2) * d[24] +
                  2 * d[9] * d[10] * d[25] + 2 * d[12] * d[13] * d[25];
    coeffs[193] = d[15] * std::pow(d[18], 2) + 2 * d[16] * d[18] * d[19] - d[15] * std::pow(d[19], 2) +
                  d[15] * std::pow(d[21], 2) + 2 * d[16] * d[21] * d[22] - d[15] * std::pow(d[22], 2) +
                  2 * d[9] * d[18] * d[24] - 2 * d[10] * d[19] * d[24] + 2 * d[12] * d[21] * d[24] -
                  2 * d[13] * d[22] * d[24] + 2 * d[10] * d[18] * d[25] + 2 * d[9] * d[19] * d[25] +
                  2 * d[13] * d[21] * d[25] + 2 * d[12] * d[22] * d[25];
    coeffs[194] = std::pow(d[18], 2) * d[24] - std::pow(d[19], 2) * d[24] + std::pow(d[21], 2) * d[24] -
                  std::pow(d[22], 2) * d[24] + 2 * d[18] * d[19] * d[25] + 2 * d[21] * d[22] * d[25];
    coeffs[195] = 2 * d[8] * d[9] * d[11] - d[6] * std::pow(d[11], 2) + 2 * d[8] * d[12] * d[14] -
                  d[6] * std::pow(d[14], 2) - 2 * d[2] * d[11] * d[15] - 2 * d[5] * d[14] * d[15] +
                  3 * d[6] * std::pow(d[15], 2) + 2 * d[7] * d[15] * d[16] + d[6] * std::pow(d[16], 2) +
                  2 * d[2] * d[9] * d[17] + 2 * d[0] * d[11] * d[17] + 2 * d[5] * d[12] * d[17] +
                  2 * d[3] * d[14] * d[17];
    coeffs[196] =
        2 * d[8] * d[11] * d[18] + 2 * d[2] * d[17] * d[18] + 2 * d[8] * d[9] * d[20] - 2 * d[6] * d[11] * d[20] -
        2 * d[2] * d[15] * d[20] + 2 * d[0] * d[17] * d[20] + 2 * d[8] * d[14] * d[21] + 2 * d[5] * d[17] * d[21] +
        2 * d[8] * d[12] * d[23] - 2 * d[6] * d[14] * d[23] - 2 * d[5] * d[15] * d[23] + 2 * d[3] * d[17] * d[23] -
        2 * d[2] * d[11] * d[24] - 2 * d[5] * d[14] * d[24] + 6 * d[6] * d[15] * d[24] + 2 * d[7] * d[16] * d[24] +
        2 * d[7] * d[15] * d[25] + 2 * d[6] * d[16] * d[25] + 2 * d[2] * d[9] * d[26] + 2 * d[0] * d[11] * d[26] +
        2 * d[5] * d[12] * d[26] + 2 * d[3] * d[14] * d[26];
    coeffs[197] = 2 * d[8] * d[18] * d[20] - d[6] * std::pow(d[20], 2) + 2 * d[8] * d[21] * d[23] -
                  d[6] * std::pow(d[23], 2) - 2 * d[2] * d[20] * d[24] - 2 * d[5] * d[23] * d[24] +
                  3 * d[6] * std::pow(d[24], 2) + 2 * d[7] * d[24] * d[25] + d[6] * std::pow(d[25], 2) +
                  2 * d[2] * d[18] * d[26] + 2 * d[0] * d[20] * d[26] + 2 * d[5] * d[21] * d[26] +
                  2 * d[3] * d[23] * d[26];
    coeffs[198] = std::pow(d[8], 2) * d[15] + 2 * d[6] * d[8] * d[17];
    coeffs[199] = std::pow(d[8], 2) * d[24] + 2 * d[6] * d[8] * d[26];
    coeffs[200] = d[6] * std::pow(d[9], 2) + 2 * d[7] * d[9] * d[10] - d[6] * std::pow(d[10], 2) +
                  d[6] * std::pow(d[12], 2) + 2 * d[7] * d[12] * d[13] - d[6] * std::pow(d[13], 2) +
                  2 * d[0] * d[9] * d[15] - 2 * d[1] * d[10] * d[15] + 2 * d[3] * d[12] * d[15] -
                  2 * d[4] * d[13] * d[15] + 2 * d[1] * d[9] * d[16] + 2 * d[0] * d[10] * d[16] +
                  2 * d[4] * d[12] * d[16] + 2 * d[3] * d[13] * d[16];
    coeffs[201] =
        2 * d[6] * d[9] * d[18] + 2 * d[7] * d[10] * d[18] + 2 * d[0] * d[15] * d[18] + 2 * d[1] * d[16] * d[18] +
        2 * d[7] * d[9] * d[19] - 2 * d[6] * d[10] * d[19] - 2 * d[1] * d[15] * d[19] + 2 * d[0] * d[16] * d[19] +
        2 * d[6] * d[12] * d[21] + 2 * d[7] * d[13] * d[21] + 2 * d[3] * d[15] * d[21] + 2 * d[4] * d[16] * d[21] +
        2 * d[7] * d[12] * d[22] - 2 * d[6] * d[13] * d[22] - 2 * d[4] * d[15] * d[22] + 2 * d[3] * d[16] * d[22] +
        2 * d[0] * d[9] * d[24] - 2 * d[1] * d[10] * d[24] + 2 * d[3] * d[12] * d[24] - 2 * d[4] * d[13] * d[24] +
        2 * d[1] * d[9] * d[25] + 2 * d[0] * d[10] * d[25] + 2 * d[4] * d[12] * d[25] + 2 * d[3] * d[13] * d[25];
    coeffs[202] = d[6] * std::pow(d[18], 2) + 2 * d[7] * d[18] * d[19] - d[6] * std::pow(d[19], 2) +
                  d[6] * std::pow(d[21], 2) + 2 * d[7] * d[21] * d[22] - d[6] * std::pow(d[22], 2) +
                  2 * d[0] * d[18] * d[24] - 2 * d[1] * d[19] * d[24] + 2 * d[3] * d[21] * d[24] -
                  2 * d[4] * d[22] * d[24] + 2 * d[1] * d[18] * d[25] + 2 * d[0] * d[19] * d[25] +
                  2 * d[4] * d[21] * d[25] + 2 * d[3] * d[22] * d[25];
    coeffs[203] = 2 * d[2] * d[8] * d[9] - 2 * d[2] * d[6] * d[11] + 2 * d[0] * d[8] * d[11] + 2 * d[5] * d[8] * d[12] -
                  2 * d[5] * d[6] * d[14] + 2 * d[3] * d[8] * d[14] - std::pow(d[2], 2) * d[15] -
                  std::pow(d[5], 2) * d[15] + 3 * std::pow(d[6], 2) * d[15] + std::pow(d[7], 2) * d[15] +
                  2 * d[6] * d[7] * d[16] + 2 * d[0] * d[2] * d[17] + 2 * d[3] * d[5] * d[17];
    coeffs[204] = 2 * d[2] * d[8] * d[18] - 2 * d[2] * d[6] * d[20] + 2 * d[0] * d[8] * d[20] +
                  2 * d[5] * d[8] * d[21] - 2 * d[5] * d[6] * d[23] + 2 * d[3] * d[8] * d[23] -
                  std::pow(d[2], 2) * d[24] - std::pow(d[5], 2) * d[24] + 3 * std::pow(d[6], 2) * d[24] +
                  std::pow(d[7], 2) * d[24] + 2 * d[6] * d[7] * d[25] + 2 * d[0] * d[2] * d[26] +
                  2 * d[3] * d[5] * d[26];
    coeffs[205] = d[6] * std::pow(d[8], 2);
    coeffs[206] = 2 * d[0] * d[6] * d[9] + 2 * d[1] * d[7] * d[9] - 2 * d[1] * d[6] * d[10] + 2 * d[0] * d[7] * d[10] +
                  2 * d[3] * d[6] * d[12] + 2 * d[4] * d[7] * d[12] - 2 * d[4] * d[6] * d[13] +
                  2 * d[3] * d[7] * d[13] + std::pow(d[0], 2) * d[15] - std::pow(d[1], 2) * d[15] +
                  std::pow(d[3], 2) * d[15] - std::pow(d[4], 2) * d[15] + 2 * d[0] * d[1] * d[16] +
                  2 * d[3] * d[4] * d[16];
    coeffs[207] = 2 * d[0] * d[6] * d[18] + 2 * d[1] * d[7] * d[18] - 2 * d[1] * d[6] * d[19] +
                  2 * d[0] * d[7] * d[19] + 2 * d[3] * d[6] * d[21] + 2 * d[4] * d[7] * d[21] -
                  2 * d[4] * d[6] * d[22] + 2 * d[3] * d[7] * d[22] + std::pow(d[0], 2) * d[24] -
                  std::pow(d[1], 2) * d[24] + std::pow(d[3], 2) * d[24] - std::pow(d[4], 2) * d[24] +
                  2 * d[0] * d[1] * d[25] + 2 * d[3] * d[4] * d[25];
    coeffs[208] = -std::pow(d[2], 2) * d[6] - std::pow(d[5], 2) * d[6] + std::pow(d[6], 3) + d[6] * std::pow(d[7], 2) +
                  2 * d[0] * d[2] * d[8] + 2 * d[3] * d[5] * d[8];
    coeffs[209] = std::pow(d[0], 2) * d[6] - std::pow(d[1], 2) * d[6] + std::pow(d[3], 2) * d[6] -
                  std::pow(d[4], 2) * d[6] + 2 * d[0] * d[1] * d[7] + 2 * d[3] * d[4] * d[7];
    coeffs[210] = d[16] * std::pow(d[17], 2);
    coeffs[211] = std::pow(d[17], 2) * d[25] + 2 * d[16] * d[17] * d[26];
    coeffs[212] = 2 * d[17] * d[25] * d[26] + d[16] * std::pow(d[26], 2);
    coeffs[213] = d[25] * std::pow(d[26], 2);
    coeffs[214] = -std::pow(d[11], 2) * d[16] - std::pow(d[14], 2) * d[16] + std::pow(d[15], 2) * d[16] +
                  std::pow(d[16], 3) + 2 * d[10] * d[11] * d[17] + 2 * d[13] * d[14] * d[17];
    coeffs[215] = 2 * d[11] * d[17] * d[19] - 2 * d[11] * d[16] * d[20] + 2 * d[10] * d[17] * d[20] +
                  2 * d[14] * d[17] * d[22] - 2 * d[14] * d[16] * d[23] + 2 * d[13] * d[17] * d[23] +
                  2 * d[15] * d[16] * d[24] - std::pow(d[11], 2) * d[25] - std::pow(d[14], 2) * d[25] +
                  std::pow(d[15], 2) * d[25] + 3 * std::pow(d[16], 2) * d[25] + 2 * d[10] * d[11] * d[26] +
                  2 * d[13] * d[14] * d[26];
    coeffs[216] = 2 * d[17] * d[19] * d[20] - d[16] * std::pow(d[20], 2) + 2 * d[17] * d[22] * d[23] -
                  d[16] * std::pow(d[23], 2) + d[16] * std::pow(d[24], 2) - 2 * d[11] * d[20] * d[25] -
                  2 * d[14] * d[23] * d[25] + 2 * d[15] * d[24] * d[25] + 3 * d[16] * std::pow(d[25], 2) +
                  2 * d[11] * d[19] * d[26] + 2 * d[10] * d[20] * d[26] + 2 * d[14] * d[22] * d[26] +
                  2 * d[13] * d[23] * d[26];
    coeffs[217] = -std::pow(d[20], 2) * d[25] - std::pow(d[23], 2) * d[25] + std::pow(d[24], 2) * d[25] +
                  std::pow(d[25], 3) + 2 * d[19] * d[20] * d[26] + 2 * d[22] * d[23] * d[26];
    coeffs[218] = 2 * d[8] * d[16] * d[17] + d[7] * std::pow(d[17], 2);
    coeffs[219] = 2 * d[8] * d[17] * d[25] + 2 * d[8] * d[16] * d[26] + 2 * d[7] * d[17] * d[26];
    coeffs[220] = 2 * d[8] * d[25] * d[26] + d[7] * std::pow(d[26], 2);
    coeffs[221] = 2 * d[9] * d[10] * d[15] + 2 * d[12] * d[13] * d[15] - std::pow(d[9], 2) * d[16] +
                  std::pow(d[10], 2) * d[16] - std::pow(d[12], 2) * d[16] + std::pow(d[13], 2) * d[16];
    coeffs[222] = 2 * d[10] * d[15] * d[18] - 2 * d[9] * d[16] * d[18] + 2 * d[9] * d[15] * d[19] +
                  2 * d[10] * d[16] * d[19] + 2 * d[13] * d[15] * d[21] - 2 * d[12] * d[16] * d[21] +
                  2 * d[12] * d[15] * d[22] + 2 * d[13] * d[16] * d[22] + 2 * d[9] * d[10] * d[24] +
                  2 * d[12] * d[13] * d[24] - std::pow(d[9], 2) * d[25] + std::pow(d[10], 2) * d[25] -
                  std::pow(d[12], 2) * d[25] + std::pow(d[13], 2) * d[25];
    coeffs[223] = -d[16] * std::pow(d[18], 2) + 2 * d[15] * d[18] * d[19] + d[16] * std::pow(d[19], 2) -
                  d[16] * std::pow(d[21], 2) + 2 * d[15] * d[21] * d[22] + d[16] * std::pow(d[22], 2) +
                  2 * d[10] * d[18] * d[24] + 2 * d[9] * d[19] * d[24] + 2 * d[13] * d[21] * d[24] +
                  2 * d[12] * d[22] * d[24] - 2 * d[9] * d[18] * d[25] + 2 * d[10] * d[19] * d[25] -
                  2 * d[12] * d[21] * d[25] + 2 * d[13] * d[22] * d[25];
    coeffs[224] = 2 * d[18] * d[19] * d[24] + 2 * d[21] * d[22] * d[24] - std::pow(d[18], 2) * d[25] +
                  std::pow(d[19], 2) * d[25] - std::pow(d[21], 2) * d[25] + std::pow(d[22], 2) * d[25];
    coeffs[225] = 2 * d[8] * d[10] * d[11] - d[7] * std::pow(d[11], 2) + 2 * d[8] * d[13] * d[14] -
                  d[7] * std::pow(d[14], 2) + d[7] * std::pow(d[15], 2) - 2 * d[2] * d[11] * d[16] -
                  2 * d[5] * d[14] * d[16] + 2 * d[6] * d[15] * d[16] + 3 * d[7] * std::pow(d[16], 2) +
                  2 * d[2] * d[10] * d[17] + 2 * d[1] * d[11] * d[17] + 2 * d[5] * d[13] * d[17] +
                  2 * d[4] * d[14] * d[17];
    coeffs[226] =
        2 * d[8] * d[11] * d[19] + 2 * d[2] * d[17] * d[19] + 2 * d[8] * d[10] * d[20] - 2 * d[7] * d[11] * d[20] -
        2 * d[2] * d[16] * d[20] + 2 * d[1] * d[17] * d[20] + 2 * d[8] * d[14] * d[22] + 2 * d[5] * d[17] * d[22] +
        2 * d[8] * d[13] * d[23] - 2 * d[7] * d[14] * d[23] - 2 * d[5] * d[16] * d[23] + 2 * d[4] * d[17] * d[23] +
        2 * d[7] * d[15] * d[24] + 2 * d[6] * d[16] * d[24] - 2 * d[2] * d[11] * d[25] - 2 * d[5] * d[14] * d[25] +
        2 * d[6] * d[15] * d[25] + 6 * d[7] * d[16] * d[25] + 2 * d[2] * d[10] * d[26] + 2 * d[1] * d[11] * d[26] +
        2 * d[5] * d[13] * d[26] + 2 * d[4] * d[14] * d[26];
    coeffs[227] = 2 * d[8] * d[19] * d[20] - d[7] * std::pow(d[20], 2) + 2 * d[8] * d[22] * d[23] -
                  d[7] * std::pow(d[23], 2) + d[7] * std::pow(d[24], 2) - 2 * d[2] * d[20] * d[25] -
                  2 * d[5] * d[23] * d[25] + 2 * d[6] * d[24] * d[25] + 3 * d[7] * std::pow(d[25], 2) +
                  2 * d[2] * d[19] * d[26] + 2 * d[1] * d[20] * d[26] + 2 * d[5] * d[22] * d[26] +
                  2 * d[4] * d[23] * d[26];
    coeffs[228] = std::pow(d[8], 2) * d[16] + 2 * d[7] * d[8] * d[17];
    coeffs[229] = std::pow(d[8], 2) * d[25] + 2 * d[7] * d[8] * d[26];
    coeffs[230] = -d[7] * std::pow(d[9], 2) + 2 * d[6] * d[9] * d[10] + d[7] * std::pow(d[10], 2) -
                  d[7] * std::pow(d[12], 2) + 2 * d[6] * d[12] * d[13] + d[7] * std::pow(d[13], 2) +
                  2 * d[1] * d[9] * d[15] + 2 * d[0] * d[10] * d[15] + 2 * d[4] * d[12] * d[15] +
                  2 * d[3] * d[13] * d[15] - 2 * d[0] * d[9] * d[16] + 2 * d[1] * d[10] * d[16] -
                  2 * d[3] * d[12] * d[16] + 2 * d[4] * d[13] * d[16];
    coeffs[231] =
        -2 * d[7] * d[9] * d[18] + 2 * d[6] * d[10] * d[18] + 2 * d[1] * d[15] * d[18] - 2 * d[0] * d[16] * d[18] +
        2 * d[6] * d[9] * d[19] + 2 * d[7] * d[10] * d[19] + 2 * d[0] * d[15] * d[19] + 2 * d[1] * d[16] * d[19] -
        2 * d[7] * d[12] * d[21] + 2 * d[6] * d[13] * d[21] + 2 * d[4] * d[15] * d[21] - 2 * d[3] * d[16] * d[21] +
        2 * d[6] * d[12] * d[22] + 2 * d[7] * d[13] * d[22] + 2 * d[3] * d[15] * d[22] + 2 * d[4] * d[16] * d[22] +
        2 * d[1] * d[9] * d[24] + 2 * d[0] * d[10] * d[24] + 2 * d[4] * d[12] * d[24] + 2 * d[3] * d[13] * d[24] -
        2 * d[0] * d[9] * d[25] + 2 * d[1] * d[10] * d[25] - 2 * d[3] * d[12] * d[25] + 2 * d[4] * d[13] * d[25];
    coeffs[232] = -d[7] * std::pow(d[18], 2) + 2 * d[6] * d[18] * d[19] + d[7] * std::pow(d[19], 2) -
                  d[7] * std::pow(d[21], 2) + 2 * d[6] * d[21] * d[22] + d[7] * std::pow(d[22], 2) +
                  2 * d[1] * d[18] * d[24] + 2 * d[0] * d[19] * d[24] + 2 * d[4] * d[21] * d[24] +
                  2 * d[3] * d[22] * d[24] - 2 * d[0] * d[18] * d[25] + 2 * d[1] * d[19] * d[25] -
                  2 * d[3] * d[21] * d[25] + 2 * d[4] * d[22] * d[25];
    coeffs[233] = 2 * d[2] * d[8] * d[10] - 2 * d[2] * d[7] * d[11] + 2 * d[1] * d[8] * d[11] +
                  2 * d[5] * d[8] * d[13] - 2 * d[5] * d[7] * d[14] + 2 * d[4] * d[8] * d[14] +
                  2 * d[6] * d[7] * d[15] - std::pow(d[2], 2) * d[16] - std::pow(d[5], 2) * d[16] +
                  std::pow(d[6], 2) * d[16] + 3 * std::pow(d[7], 2) * d[16] + 2 * d[1] * d[2] * d[17] +
                  2 * d[4] * d[5] * d[17];
    coeffs[234] = 2 * d[2] * d[8] * d[19] - 2 * d[2] * d[7] * d[20] + 2 * d[1] * d[8] * d[20] +
                  2 * d[5] * d[8] * d[22] - 2 * d[5] * d[7] * d[23] + 2 * d[4] * d[8] * d[23] +
                  2 * d[6] * d[7] * d[24] - std::pow(d[2], 2) * d[25] - std::pow(d[5], 2) * d[25] +
                  std::pow(d[6], 2) * d[25] + 3 * std::pow(d[7], 2) * d[25] + 2 * d[1] * d[2] * d[26] +
                  2 * d[4] * d[5] * d[26];
    coeffs[235] = d[7] * std::pow(d[8], 2);
    coeffs[236] = 2 * d[1] * d[6] * d[9] - 2 * d[0] * d[7] * d[9] + 2 * d[0] * d[6] * d[10] + 2 * d[1] * d[7] * d[10] +
                  2 * d[4] * d[6] * d[12] - 2 * d[3] * d[7] * d[12] + 2 * d[3] * d[6] * d[13] +
                  2 * d[4] * d[7] * d[13] + 2 * d[0] * d[1] * d[15] + 2 * d[3] * d[4] * d[15] -
                  std::pow(d[0], 2) * d[16] + std::pow(d[1], 2) * d[16] - std::pow(d[3], 2) * d[16] +
                  std::pow(d[4], 2) * d[16];
    coeffs[237] = 2 * d[1] * d[6] * d[18] - 2 * d[0] * d[7] * d[18] + 2 * d[0] * d[6] * d[19] +
                  2 * d[1] * d[7] * d[19] + 2 * d[4] * d[6] * d[21] - 2 * d[3] * d[7] * d[21] +
                  2 * d[3] * d[6] * d[22] + 2 * d[4] * d[7] * d[22] + 2 * d[0] * d[1] * d[24] +
                  2 * d[3] * d[4] * d[24] - std::pow(d[0], 2) * d[25] + std::pow(d[1], 2) * d[25] -
                  std::pow(d[3], 2) * d[25] + std::pow(d[4], 2) * d[25];
    coeffs[238] = -std::pow(d[2], 2) * d[7] - std::pow(d[5], 2) * d[7] + std::pow(d[6], 2) * d[7] + std::pow(d[7], 3) +
                  2 * d[1] * d[2] * d[8] + 2 * d[4] * d[5] * d[8];
    coeffs[239] = 2 * d[0] * d[1] * d[6] + 2 * d[3] * d[4] * d[6] - std::pow(d[0], 2) * d[7] +
                  std::pow(d[1], 2) * d[7] - std::pow(d[3], 2) * d[7] + std::pow(d[4], 2) * d[7];
    coeffs[240] = std::pow(d[17], 3);
    coeffs[241] = 3 * std::pow(d[17], 2) * d[26];
    coeffs[242] = 3 * d[17] * std::pow(d[26], 2);
    coeffs[243] = std::pow(d[26], 3);
    coeffs[244] = std::pow(d[11], 2) * d[17] + std::pow(d[14], 2) * d[17] + std::pow(d[15], 2) * d[17] +
                  std::pow(d[16], 2) * d[17];
    coeffs[245] = 2 * d[11] * d[17] * d[20] + 2 * d[14] * d[17] * d[23] + 2 * d[15] * d[17] * d[24] +
                  2 * d[16] * d[17] * d[25] + std::pow(d[11], 2) * d[26] + std::pow(d[14], 2) * d[26] +
                  std::pow(d[15], 2) * d[26] + std::pow(d[16], 2) * d[26];
    coeffs[246] = d[17] * std::pow(d[20], 2) + d[17] * std::pow(d[23], 2) + d[17] * std::pow(d[24], 2) +
                  d[17] * std::pow(d[25], 2) + 2 * d[11] * d[20] * d[26] + 2 * d[14] * d[23] * d[26] +
                  2 * d[15] * d[24] * d[26] + 2 * d[16] * d[25] * d[26];
    coeffs[247] = std::pow(d[20], 2) * d[26] + std::pow(d[23], 2) * d[26] + std::pow(d[24], 2) * d[26] +
                  std::pow(d[25], 2) * d[26];
    coeffs[248] = 3 * d[8] * std::pow(d[17], 2);
    coeffs[249] = 6 * d[8] * d[17] * d[26];
    coeffs[250] = 3 * d[8] * std::pow(d[26], 2);
    coeffs[251] = 2 * d[9] * d[11] * d[15] + 2 * d[12] * d[14] * d[15] + 2 * d[10] * d[11] * d[16] +
                  2 * d[13] * d[14] * d[16] - std::pow(d[9], 2) * d[17] - std::pow(d[10], 2) * d[17] -
                  std::pow(d[12], 2) * d[17] - std::pow(d[13], 2) * d[17];
    coeffs[252] = 2 * d[11] * d[15] * d[18] - 2 * d[9] * d[17] * d[18] + 2 * d[11] * d[16] * d[19] -
                  2 * d[10] * d[17] * d[19] + 2 * d[9] * d[15] * d[20] + 2 * d[10] * d[16] * d[20] +
                  2 * d[14] * d[15] * d[21] - 2 * d[12] * d[17] * d[21] + 2 * d[14] * d[16] * d[22] -
                  2 * d[13] * d[17] * d[22] + 2 * d[12] * d[15] * d[23] + 2 * d[13] * d[16] * d[23] +
                  2 * d[9] * d[11] * d[24] + 2 * d[12] * d[14] * d[24] + 2 * d[10] * d[11] * d[25] +
                  2 * d[13] * d[14] * d[25] - std::pow(d[9], 2) * d[26] - std::pow(d[10], 2) * d[26] -
                  std::pow(d[12], 2) * d[26] - std::pow(d[13], 2) * d[26];
    coeffs[253] = -d[17] * std::pow(d[18], 2) - d[17] * std::pow(d[19], 2) + 2 * d[15] * d[18] * d[20] +
                  2 * d[16] * d[19] * d[20] - d[17] * std::pow(d[21], 2) - d[17] * std::pow(d[22], 2) +
                  2 * d[15] * d[21] * d[23] + 2 * d[16] * d[22] * d[23] + 2 * d[11] * d[18] * d[24] +
                  2 * d[9] * d[20] * d[24] + 2 * d[14] * d[21] * d[24] + 2 * d[12] * d[23] * d[24] +
                  2 * d[11] * d[19] * d[25] + 2 * d[10] * d[20] * d[25] + 2 * d[14] * d[22] * d[25] +
                  2 * d[13] * d[23] * d[25] - 2 * d[9] * d[18] * d[26] - 2 * d[10] * d[19] * d[26] -
                  2 * d[12] * d[21] * d[26] - 2 * d[13] * d[22] * d[26];
    coeffs[254] = 2 * d[18] * d[20] * d[24] + 2 * d[21] * d[23] * d[24] + 2 * d[19] * d[20] * d[25] +
                  2 * d[22] * d[23] * d[25] - std::pow(d[18], 2) * d[26] - std::pow(d[19], 2) * d[26] -
                  std::pow(d[21], 2) * d[26] - std::pow(d[22], 2) * d[26];
    coeffs[255] = d[8] * std::pow(d[11], 2) + d[8] * std::pow(d[14], 2) + d[8] * std::pow(d[15], 2) +
                  d[8] * std::pow(d[16], 2) + 2 * d[2] * d[11] * d[17] + 2 * d[5] * d[14] * d[17] +
                  2 * d[6] * d[15] * d[17] + 2 * d[7] * d[16] * d[17];
    coeffs[256] = 2 * d[8] * d[11] * d[20] + 2 * d[2] * d[17] * d[20] + 2 * d[8] * d[14] * d[23] +
                  2 * d[5] * d[17] * d[23] + 2 * d[8] * d[15] * d[24] + 2 * d[6] * d[17] * d[24] +
                  2 * d[8] * d[16] * d[25] + 2 * d[7] * d[17] * d[25] + 2 * d[2] * d[11] * d[26] +
                  2 * d[5] * d[14] * d[26] + 2 * d[6] * d[15] * d[26] + 2 * d[7] * d[16] * d[26];
    coeffs[257] = d[8] * std::pow(d[20], 2) + d[8] * std::pow(d[23], 2) + d[8] * std::pow(d[24], 2) +
                  d[8] * std::pow(d[25], 2) + 2 * d[2] * d[20] * d[26] + 2 * d[5] * d[23] * d[26] +
                  2 * d[6] * d[24] * d[26] + 2 * d[7] * d[25] * d[26];
    coeffs[258] = 3 * std::pow(d[8], 2) * d[17];
    coeffs[259] = 3 * std::pow(d[8], 2) * d[26];
    coeffs[260] =
        -d[8] * std::pow(d[9], 2) - d[8] * std::pow(d[10], 2) + 2 * d[6] * d[9] * d[11] + 2 * d[7] * d[10] * d[11] -
        d[8] * std::pow(d[12], 2) - d[8] * std::pow(d[13], 2) + 2 * d[6] * d[12] * d[14] + 2 * d[7] * d[13] * d[14] +
        2 * d[2] * d[9] * d[15] + 2 * d[0] * d[11] * d[15] + 2 * d[5] * d[12] * d[15] + 2 * d[3] * d[14] * d[15] +
        2 * d[2] * d[10] * d[16] + 2 * d[1] * d[11] * d[16] + 2 * d[5] * d[13] * d[16] + 2 * d[4] * d[14] * d[16] -
        2 * d[0] * d[9] * d[17] - 2 * d[1] * d[10] * d[17] - 2 * d[3] * d[12] * d[17] - 2 * d[4] * d[13] * d[17];
    coeffs[261] =
        -2 * d[8] * d[9] * d[18] + 2 * d[6] * d[11] * d[18] + 2 * d[2] * d[15] * d[18] - 2 * d[0] * d[17] * d[18] -
        2 * d[8] * d[10] * d[19] + 2 * d[7] * d[11] * d[19] + 2 * d[2] * d[16] * d[19] - 2 * d[1] * d[17] * d[19] +
        2 * d[6] * d[9] * d[20] + 2 * d[7] * d[10] * d[20] + 2 * d[0] * d[15] * d[20] + 2 * d[1] * d[16] * d[20] -
        2 * d[8] * d[12] * d[21] + 2 * d[6] * d[14] * d[21] + 2 * d[5] * d[15] * d[21] - 2 * d[3] * d[17] * d[21] -
        2 * d[8] * d[13] * d[22] + 2 * d[7] * d[14] * d[22] + 2 * d[5] * d[16] * d[22] - 2 * d[4] * d[17] * d[22] +
        2 * d[6] * d[12] * d[23] + 2 * d[7] * d[13] * d[23] + 2 * d[3] * d[15] * d[23] + 2 * d[4] * d[16] * d[23] +
        2 * d[2] * d[9] * d[24] + 2 * d[0] * d[11] * d[24] + 2 * d[5] * d[12] * d[24] + 2 * d[3] * d[14] * d[24] +
        2 * d[2] * d[10] * d[25] + 2 * d[1] * d[11] * d[25] + 2 * d[5] * d[13] * d[25] + 2 * d[4] * d[14] * d[25] -
        2 * d[0] * d[9] * d[26] - 2 * d[1] * d[10] * d[26] - 2 * d[3] * d[12] * d[26] - 2 * d[4] * d[13] * d[26];
    coeffs[262] =
        -d[8] * std::pow(d[18], 2) - d[8] * std::pow(d[19], 2) + 2 * d[6] * d[18] * d[20] + 2 * d[7] * d[19] * d[20] -
        d[8] * std::pow(d[21], 2) - d[8] * std::pow(d[22], 2) + 2 * d[6] * d[21] * d[23] + 2 * d[7] * d[22] * d[23] +
        2 * d[2] * d[18] * d[24] + 2 * d[0] * d[20] * d[24] + 2 * d[5] * d[21] * d[24] + 2 * d[3] * d[23] * d[24] +
        2 * d[2] * d[19] * d[25] + 2 * d[1] * d[20] * d[25] + 2 * d[5] * d[22] * d[25] + 2 * d[4] * d[23] * d[25] -
        2 * d[0] * d[18] * d[26] - 2 * d[1] * d[19] * d[26] - 2 * d[3] * d[21] * d[26] - 2 * d[4] * d[22] * d[26];
    coeffs[263] = 2 * d[2] * d[8] * d[11] + 2 * d[5] * d[8] * d[14] + 2 * d[6] * d[8] * d[15] +
                  2 * d[7] * d[8] * d[16] + std::pow(d[2], 2) * d[17] + std::pow(d[5], 2) * d[17] +
                  std::pow(d[6], 2) * d[17] + std::pow(d[7], 2) * d[17];
    coeffs[264] = 2 * d[2] * d[8] * d[20] + 2 * d[5] * d[8] * d[23] + 2 * d[6] * d[8] * d[24] +
                  2 * d[7] * d[8] * d[25] + std::pow(d[2], 2) * d[26] + std::pow(d[5], 2) * d[26] +
                  std::pow(d[6], 2) * d[26] + std::pow(d[7], 2) * d[26];
    coeffs[265] = std::pow(d[8], 3);
    coeffs[266] =
        2 * d[2] * d[6] * d[9] - 2 * d[0] * d[8] * d[9] + 2 * d[2] * d[7] * d[10] - 2 * d[1] * d[8] * d[10] +
        2 * d[0] * d[6] * d[11] + 2 * d[1] * d[7] * d[11] + 2 * d[5] * d[6] * d[12] - 2 * d[3] * d[8] * d[12] +
        2 * d[5] * d[7] * d[13] - 2 * d[4] * d[8] * d[13] + 2 * d[3] * d[6] * d[14] + 2 * d[4] * d[7] * d[14] +
        2 * d[0] * d[2] * d[15] + 2 * d[3] * d[5] * d[15] + 2 * d[1] * d[2] * d[16] + 2 * d[4] * d[5] * d[16] -
        std::pow(d[0], 2) * d[17] - std::pow(d[1], 2) * d[17] - std::pow(d[3], 2) * d[17] - std::pow(d[4], 2) * d[17];
    coeffs[267] =
        2 * d[2] * d[6] * d[18] - 2 * d[0] * d[8] * d[18] + 2 * d[2] * d[7] * d[19] - 2 * d[1] * d[8] * d[19] +
        2 * d[0] * d[6] * d[20] + 2 * d[1] * d[7] * d[20] + 2 * d[5] * d[6] * d[21] - 2 * d[3] * d[8] * d[21] +
        2 * d[5] * d[7] * d[22] - 2 * d[4] * d[8] * d[22] + 2 * d[3] * d[6] * d[23] + 2 * d[4] * d[7] * d[23] +
        2 * d[0] * d[2] * d[24] + 2 * d[3] * d[5] * d[24] + 2 * d[1] * d[2] * d[25] + 2 * d[4] * d[5] * d[25] -
        std::pow(d[0], 2) * d[26] - std::pow(d[1], 2) * d[26] - std::pow(d[3], 2) * d[26] - std::pow(d[4], 2) * d[26];
    coeffs[268] =
        std::pow(d[2], 2) * d[8] + std::pow(d[5], 2) * d[8] + std::pow(d[6], 2) * d[8] + std::pow(d[7], 2) * d[8];
    coeffs[269] = 2 * d[0] * d[2] * d[6] + 2 * d[3] * d[5] * d[6] + 2 * d[1] * d[2] * d[7] + 2 * d[4] * d[5] * d[7] -
                  std::pow(d[0], 2) * d[8] - std::pow(d[1], 2) * d[8] - std::pow(d[3], 2) * d[8] -
                  std::pow(d[4], 2) * d[8];
    coeffs[270] = -d[11] * d[13] * d[15] + d[10] * d[14] * d[15] + d[11] * d[12] * d[16] - d[9] * d[14] * d[16] -
                  d[10] * d[12] * d[17] + d[9] * d[13] * d[17];
    coeffs[271] = -d[14] * d[16] * d[18] + d[13] * d[17] * d[18] + d[14] * d[15] * d[19] - d[12] * d[17] * d[19] -
                  d[13] * d[15] * d[20] + d[12] * d[16] * d[20] + d[11] * d[16] * d[21] - d[10] * d[17] * d[21] -
                  d[11] * d[15] * d[22] + d[9] * d[17] * d[22] + d[10] * d[15] * d[23] - d[9] * d[16] * d[23] -
                  d[11] * d[13] * d[24] + d[10] * d[14] * d[24] + d[11] * d[12] * d[25] - d[9] * d[14] * d[25] -
                  d[10] * d[12] * d[26] + d[9] * d[13] * d[26];
    coeffs[272] = -d[17] * d[19] * d[21] + d[16] * d[20] * d[21] + d[17] * d[18] * d[22] - d[15] * d[20] * d[22] -
                  d[16] * d[18] * d[23] + d[15] * d[19] * d[23] + d[14] * d[19] * d[24] - d[13] * d[20] * d[24] -
                  d[11] * d[22] * d[24] + d[10] * d[23] * d[24] - d[14] * d[18] * d[25] + d[12] * d[20] * d[25] +
                  d[11] * d[21] * d[25] - d[9] * d[23] * d[25] + d[13] * d[18] * d[26] - d[12] * d[19] * d[26] -
                  d[10] * d[21] * d[26] + d[9] * d[22] * d[26];
    coeffs[273] = -d[20] * d[22] * d[24] + d[19] * d[23] * d[24] + d[20] * d[21] * d[25] - d[18] * d[23] * d[25] -
                  d[19] * d[21] * d[26] + d[18] * d[22] * d[26];
    coeffs[274] = -d[8] * d[10] * d[12] + d[7] * d[11] * d[12] + d[8] * d[9] * d[13] - d[6] * d[11] * d[13] -
                  d[7] * d[9] * d[14] + d[6] * d[10] * d[14] + d[5] * d[10] * d[15] - d[4] * d[11] * d[15] -
                  d[2] * d[13] * d[15] + d[1] * d[14] * d[15] - d[5] * d[9] * d[16] + d[3] * d[11] * d[16] +
                  d[2] * d[12] * d[16] - d[0] * d[14] * d[16] + d[4] * d[9] * d[17] - d[3] * d[10] * d[17] -
                  d[1] * d[12] * d[17] + d[0] * d[13] * d[17];
    coeffs[275] = d[8] * d[13] * d[18] - d[7] * d[14] * d[18] - d[5] * d[16] * d[18] + d[4] * d[17] * d[18] -
                  d[8] * d[12] * d[19] + d[6] * d[14] * d[19] + d[5] * d[15] * d[19] - d[3] * d[17] * d[19] +
                  d[7] * d[12] * d[20] - d[6] * d[13] * d[20] - d[4] * d[15] * d[20] + d[3] * d[16] * d[20] -
                  d[8] * d[10] * d[21] + d[7] * d[11] * d[21] + d[2] * d[16] * d[21] - d[1] * d[17] * d[21] +
                  d[8] * d[9] * d[22] - d[6] * d[11] * d[22] - d[2] * d[15] * d[22] + d[0] * d[17] * d[22] -
                  d[7] * d[9] * d[23] + d[6] * d[10] * d[23] + d[1] * d[15] * d[23] - d[0] * d[16] * d[23] +
                  d[5] * d[10] * d[24] - d[4] * d[11] * d[24] - d[2] * d[13] * d[24] + d[1] * d[14] * d[24] -
                  d[5] * d[9] * d[25] + d[3] * d[11] * d[25] + d[2] * d[12] * d[25] - d[0] * d[14] * d[25] +
                  d[4] * d[9] * d[26] - d[3] * d[10] * d[26] - d[1] * d[12] * d[26] + d[0] * d[13] * d[26];
    coeffs[276] = -d[8] * d[19] * d[21] + d[7] * d[20] * d[21] + d[8] * d[18] * d[22] - d[6] * d[20] * d[22] -
                  d[7] * d[18] * d[23] + d[6] * d[19] * d[23] + d[5] * d[19] * d[24] - d[4] * d[20] * d[24] -
                  d[2] * d[22] * d[24] + d[1] * d[23] * d[24] - d[5] * d[18] * d[25] + d[3] * d[20] * d[25] +
                  d[2] * d[21] * d[25] - d[0] * d[23] * d[25] + d[4] * d[18] * d[26] - d[3] * d[19] * d[26] -
                  d[1] * d[21] * d[26] + d[0] * d[22] * d[26];
    coeffs[277] = -d[5] * d[7] * d[9] + d[4] * d[8] * d[9] + d[5] * d[6] * d[10] - d[3] * d[8] * d[10] -
                  d[4] * d[6] * d[11] + d[3] * d[7] * d[11] + d[2] * d[7] * d[12] - d[1] * d[8] * d[12] -
                  d[2] * d[6] * d[13] + d[0] * d[8] * d[13] + d[1] * d[6] * d[14] - d[0] * d[7] * d[14] -
                  d[2] * d[4] * d[15] + d[1] * d[5] * d[15] + d[2] * d[3] * d[16] - d[0] * d[5] * d[16] -
                  d[1] * d[3] * d[17] + d[0] * d[4] * d[17];
    coeffs[278] = -d[5] * d[7] * d[18] + d[4] * d[8] * d[18] + d[5] * d[6] * d[19] - d[3] * d[8] * d[19] -
                  d[4] * d[6] * d[20] + d[3] * d[7] * d[20] + d[2] * d[7] * d[21] - d[1] * d[8] * d[21] -
                  d[2] * d[6] * d[22] + d[0] * d[8] * d[22] + d[1] * d[6] * d[23] - d[0] * d[7] * d[23] -
                  d[2] * d[4] * d[24] + d[1] * d[5] * d[24] + d[2] * d[3] * d[25] - d[0] * d[5] * d[25] -
                  d[1] * d[3] * d[26] + d[0] * d[4] * d[26];
    coeffs[279] = -d[2] * d[4] * d[6] + d[1] * d[5] * d[6] + d[2] * d[3] * d[7] - d[0] * d[5] * d[7] -
                  d[1] * d[3] * d[8] + d[0] * d[4] * d[8];

    // Setup elimination template
    static const int coeffs0_ind[] = {
        0,   30,  60,  90,  120, 150, 180, 210, 240, 1,   31,  61,  91,  121, 151, 181, 211, 241, 2,   32,  62,  92,
        122, 152, 182, 212, 242, 4,   34,  64,  30,  0,   60,  94,  124, 154, 90,  120, 150, 184, 214, 180, 210, 240,
        244, 270, 5,   35,  65,  31,  1,   61,  95,  125, 155, 91,  121, 151, 185, 215, 181, 211, 241, 245, 271, 6,
        36,  66,  32,  2,   62,  96,  126, 156, 92,  122, 152, 186, 216, 182, 212, 242, 246, 272, 7,   37,  67,  33,
        3,   63,  97,  127, 157, 93,  123, 153, 187, 217, 183, 213, 243, 247, 273, 8,   38,  68,  98,  128, 158, 188,
        218, 248, 9,   39,  69,  99,  129, 159, 189, 219, 249, 11,  41,  71,  34,  4,   64,  101, 131, 161, 90,  94,
        60,  120, 124, 154, 191, 221, 0,   180, 184, 150, 210, 214, 30,  240, 244, 251, 270, 12,  42,  72,  35,  5,
        65,  102, 132, 162, 91,  95,  61,  121, 125, 155, 192, 222, 1,   181, 185, 151, 211, 215, 31,  241, 245, 252,
        271, 13,  43,  73,  36,  6,   66,  103, 133, 163, 92,  96,  62,  122, 126, 156, 193, 223, 2,   182, 186, 152,
        212, 216, 32,  242, 246, 253, 272, 14,  44,  74,  37,  7,   67,  104, 134, 164, 93,  97,  63,  123, 127, 157,
        194, 224, 3,   183, 187, 153, 213, 217, 33,  243, 247, 254, 273, 15,  45,  75,  38,  8,   68,  105, 135, 165,
        98,  128, 158, 195, 225, 188, 218, 248, 255, 274, 16,  46,  76,  39,  9,   69,  106, 136, 166, 99,  129, 159,
        196, 226, 189, 219, 249, 256, 275, 17,  47,  77,  40,  10,  70,  107, 137, 167, 100, 130, 160, 197, 227, 190,
        220, 250, 257, 276, 18,  48,  78,  108, 138, 168, 198, 228, 258, 41,  11,  71,  94,  101, 64,  124, 131, 161,
        4,   184, 191, 154, 214, 221, 34,  244, 251, 270, 42,  12,  72,  95,  102, 65,  125, 132, 162, 5,   185, 192,
        155, 215, 222, 35,  245, 252, 271, 20,  50,  80,  45,  15,  75,  110, 140, 170, 98,  105, 68,  128, 135, 165,
        200, 230, 8,   188, 195, 158, 218, 225, 38,  248, 255, 260, 274, 23,  53,  83,  48,  18,  78,  113, 143, 173,
        108, 138, 168, 203, 233, 198, 228, 258, 263, 277, 101, 71,  131, 11,  191, 161, 221, 41,  251, 270, 50,  20,
        80,  105, 110, 75,  135, 140, 170, 15,  195, 200, 165, 225, 230, 45,  255, 260, 274, 102, 72,  132, 12,  192,
        162, 222, 42,  252, 271, 103, 73,  133, 13,  193, 163, 223, 43,  253, 272, 43,  13,  73,  96,  103, 66,  126,
        133, 163, 6,   186, 193, 156, 216, 223, 36,  246, 253, 272, 21,  51,  81,  46,  16,  76,  111, 141, 171, 99,
        106, 69,  129, 136, 166, 201, 231, 9,   189, 196, 159, 219, 226, 39,  249, 256, 261, 275, 104, 74,  134, 14,
        194, 164, 224, 44,  254, 273, 44,  14,  74,  97,  104, 67,  127, 134, 164, 7,   187, 194, 157, 217, 224, 37,
        247, 254, 273, 22,  52,  82,  47,  17,  77,  112, 142, 172, 100, 107, 70,  130, 137, 167, 202, 232, 10,  190,
        197, 160, 220, 227, 40,  250, 257, 262, 276, 24,  54,  84,  49,  19,  79,  114, 144, 174, 109, 139, 169, 204,
        234, 199, 229, 259, 264, 278};
    static const int coeffs1_ind[] = {
        119, 89,  149, 29,  209, 179, 239, 59,  269, 279, 116, 86,  146, 26,  206, 176, 236, 56,  266, 277, 110, 80,
        140, 20,  200, 170, 230, 50,  260, 274, 111, 81,  141, 21,  201, 171, 231, 51,  261, 275, 51,  21,  81,  106,
        111, 76,  136, 141, 171, 16,  196, 201, 166, 226, 231, 46,  256, 261, 275, 56,  26,  86,  113, 116, 83,  143,
        146, 176, 23,  203, 206, 173, 233, 236, 53,  263, 266, 277, 26,  56,  86,  53,  23,  83,  116, 146, 176, 108,
        113, 78,  138, 143, 173, 206, 236, 18,  198, 203, 168, 228, 233, 48,  258, 263, 266, 277, 117, 87,  147, 27,
        207, 177, 237, 57,  267, 278, 112, 82,  142, 22,  202, 172, 232, 52,  262, 276, 52,  22,  82,  107, 112, 77,
        137, 142, 172, 17,  197, 202, 167, 227, 232, 47,  257, 262, 276, 57,  27,  87,  114, 117, 84,  144, 147, 177,
        24,  204, 207, 174, 234, 237, 54,  264, 267, 278, 27,  57,  87,  54,  24,  84,  117, 147, 177, 109, 114, 79,
        139, 144, 174, 207, 237, 19,  199, 204, 169, 229, 234, 49,  259, 264, 267, 278, 59,  29,  89,  118, 119, 88,
        148, 149, 179, 28,  208, 209, 178, 238, 239, 58,  268, 269, 279, 29,  59,  89,  58,  28,  88,  119, 149, 179,
        115, 118, 85,  145, 148, 178, 209, 239, 25,  205, 208, 175, 235, 238, 55,  265, 268, 269, 279, 28,  58,  88,
        55,  25,  85,  118, 148, 178, 115, 145, 175, 208, 238, 205, 235, 265, 268, 279};
    static const int C0_ind[] = {
        0,   1,   2,   6,   7,   8,   15,  16,  26,  31,  32,  33,  37,  38,  39,  46,  47,  57,  62,  63,  64,  68,
        69,  70,  77,  78,  88,  93,  94,  95,  96,  97,  98,  99,  100, 101, 103, 106, 107, 108, 109, 112, 115, 118,
        119, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 134, 137, 138, 139, 140, 143, 146, 149, 150, 154, 155,
        156, 157, 158, 159, 160, 161, 162, 163, 165, 168, 169, 170, 171, 174, 177, 180, 181, 185, 186, 187, 188, 189,
        190, 191, 192, 193, 194, 196, 199, 200, 201, 202, 205, 208, 211, 212, 216, 217, 218, 219, 223, 224, 225, 232,
        233, 243, 248, 249, 250, 254, 255, 256, 263, 264, 274, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289,
        290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 308, 310, 311, 312, 313, 314,
        315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336,
        339, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361,
        362, 363, 364, 365, 366, 367, 370, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386,
        387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 401, 403, 404, 405, 406, 407, 408, 409, 410, 411,
        413, 416, 417, 418, 419, 422, 425, 428, 429, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 444, 447, 448,
        449, 450, 453, 456, 459, 460, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 475, 478, 479, 480, 481, 484,
        487, 490, 491, 495, 496, 497, 498, 502, 503, 504, 511, 512, 522, 530, 531, 532, 536, 537, 538, 539, 540, 541,
        544, 545, 546, 547, 548, 549, 550, 551, 552, 555, 561, 562, 563, 567, 568, 569, 570, 571, 572, 575, 576, 577,
        578, 579, 580, 581, 582, 583, 586, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603,
        604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 618, 620, 621, 622, 623, 624, 625, 626, 627, 628,
        630, 633, 634, 635, 636, 639, 642, 645, 646, 650, 660, 662, 663, 668, 669, 671, 672, 674, 675, 678, 685, 686,
        687, 691, 692, 693, 694, 695, 696, 699, 700, 701, 702, 703, 704, 705, 706, 707, 710, 722, 724, 725, 730, 731,
        733, 734, 736, 737, 740, 753, 755, 756, 761, 762, 764, 765, 767, 768, 771, 778, 779, 780, 784, 785, 786, 787,
        788, 789, 792, 793, 794, 795, 796, 797, 798, 799, 800, 803, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815,
        816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 835, 846, 848, 849, 854,
        855, 857, 858, 860, 861, 864, 871, 872, 873, 877, 878, 879, 880, 881, 882, 885, 886, 887, 888, 889, 890, 891,
        892, 893, 896, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917,
        918, 919, 920, 921, 922, 923, 924, 925, 928, 930, 931, 932, 933, 934, 935, 936, 937, 938, 940, 943, 944, 945,
        946, 949, 952, 955, 956, 960};
    static const int C1_ind[] = {
        9,   11,  12,  17,  18,  20,  21,  23,  24,  27,  40,  42,  43,  48,  49,  51,  52,  54,  55,  58,  71,  73,
        74,  79,  80,  82,  83,  85,  86,  89,  102, 104, 105, 110, 111, 113, 114, 116, 117, 120, 127, 128, 129, 133,
        134, 135, 136, 137, 138, 141, 142, 143, 144, 145, 146, 147, 148, 149, 152, 158, 159, 160, 164, 165, 166, 167,
        168, 169, 172, 173, 174, 175, 176, 177, 178, 179, 180, 183, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195,
        196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 215, 226, 228, 229, 234,
        235, 237, 238, 240, 241, 244, 257, 259, 260, 265, 266, 268, 269, 271, 272, 275, 282, 283, 284, 288, 289, 290,
        291, 292, 293, 296, 297, 298, 299, 300, 301, 302, 303, 304, 307, 313, 314, 315, 319, 320, 321, 322, 323, 324,
        327, 328, 329, 330, 331, 332, 333, 334, 335, 338, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352,
        353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 370, 375, 376, 377, 381, 382, 383,
        384, 385, 386, 389, 390, 391, 392, 393, 394, 395, 396, 397, 400, 403, 404, 405, 406, 407, 408, 409, 410, 411,
        412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 432, 434, 435, 436,
        437, 438, 439, 440, 441, 442, 444, 447, 448, 449, 450, 453, 456, 459, 460, 464};

    Eigen::Matrix<double, 31, 31> C0;
    C0.setZero();
    Eigen::Matrix<double, 31, 15> C1;
    C1.setZero();
    for (int i = 0; i < 556; i++) {
        C0(C0_ind[i]) = coeffs(coeffs0_ind[i]);
    }
    for (int i = 0; i < 258; i++) {
        C1(C1_ind[i]) = coeffs(coeffs1_ind[i]);
    }

    Eigen::Matrix<double, 31, 15> C12 = C0.partialPivLu().solve(C1);

    // Setup action matrix
    Eigen::Matrix<double, 23, 15> RR;
    RR << -C12.bottomRows(8), Eigen::Matrix<double, 15, 15>::Identity(15, 15);

    static const int AM_ind[] = {15, 11, 0, 1, 2, 12, 3, 16, 4, 5, 17, 6, 18, 19, 7};
    Eigen::Matrix<double, 15, 15> AM;
    for (int i = 0; i < 15; i++) {
        AM.row(i) = RR.row(AM_ind[i]);
    }

    sols.setZero();

    // Solve eigenvalue problem

    double p[1 + 15];
    Eigen::Matrix<double, 15, 15> AMp = AM;
    charpoly_danilevsky_piv(AMp, p);
    double roots[15];
    int nroots;
    find_real_roots_sturm(p, 15, roots, &nroots, 8, 0);
    fast_eigenvector_solver(roots, nroots, AM, sols);

    return nroots;
}


namespace poselib {

int relpose_6pt_singlefocal(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                            CameraOneFocaPoseVector *out_focal_poses) {

    // Compute nullspace to epipolar constraints
    Eigen::Matrix<double, 9, 6> epipolar_constraints;
    for (size_t i = 0; i < 6; ++i) {
        epipolar_constraints.col(i) << x1[i](0) * x2[i], x1[i](1) * x2[i], x1[i](2) * x2[i];
    }
    Eigen::Matrix<double, 9, 9> Q = epipolar_constraints.fullPivHouseholderQr().matrixQ();
    Eigen::Matrix<double, 9, 3> N = Q.rightCols(3);

    Eigen::VectorXd B(Eigen::Map<Eigen::VectorXd>(N.data(), N.cols() * N.rows()));

    Eigen::Matrix<std::complex<double>, 3, 15> sols;

    int n_sols = solver_relpose_6pt_singlefocal(B, sols);

    out_focal_poses->empty();

    int n_poses = 0;

    for (int i = 0; i < n_sols; i++) {
        if (sols(2, i).real() < 1e-8 || sols.col(i).imag().norm() > 1e-8) {
            continue;
        }

        double focal = std::sqrt(1.0 / sols(2, i).real());

        // std::cout << "Focal: " << focal << ", Sol: " << sols.col(i) << ", Sol img norm: " <<
        // sols.col(i).imag().norm() << "\n";

        Eigen::Vector<double, 9> F_vector = N.col(0) + sols(0, i).real() * N.col(1) + sols(1, i).real() * N.col(2);
        F_vector.normalize();
        Eigen::Matrix3d F = Eigen::Matrix3d(F_vector.data());

        // std::cout << "Inlier epipolar dist: " << x2[0].transpose() * (F * x1[0]) << "\n";

        Eigen::Matrix3d K;
        Eigen::Matrix3d K_inv;
        K << focal, 0.0, 0.0, 0.0, focal, 0.0, 0.0, 0.0, 1.0;
        K_inv << 1 / focal, 0.0, 0.0, 0.0, 1 / focal, 0.0, 0.0, 0.0, 1.0;

        Eigen::Matrix3d E = K.transpose() * (F * K);

        CameraPoseVector poses;
        motion_from_essential(E, x2[0], x1[0], &poses);

        for (CameraPose pose : poses) {
            out_focal_poses->emplace_back(CameraOneFocalPose(pose, focal));
            n_poses++;
        }
    }

    return n_poses;
}
} // namespace poselib