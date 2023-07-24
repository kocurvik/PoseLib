#include "relpose_6pt_onefocal.h"

#include <Eigen/Dense>
#include <PoseLib/misc/essential.h>
#include <iostream>
#include <math.h>
#include <stdio.h>

#define RELERROR 1.0e-12 /* smallest relative error we want */
//#define MAXPOW        0        /* max power of 10 we wish to search to */
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

    fprintf(stderr, "modrf overflow on interval %f %f\n", a, b);
    fprintf(stderr, "\t b-a = %12.5e\n", b - a);
    fprintf(stderr, "\t fa  = %12.5e\n", fa);
    fprintf(stderr, "\t fb  = %12.5e\n", fb);
    fprintf(stderr, "\t fx  = %12.5e\n", fx);

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

    fprintf(stderr, "\t true fa = %12.5e\n", fa);
    fprintf(stderr, "\t true fb = %12.5e\n", fb);
    fprintf(stderr, "\t gradient= %12.5e\n", (fb - fa) / (b - a));

    // Print out the polynomial
    fprintf(stderr, "Polynomial coefficients\n");
    for (fp = ecoef; fp >= scoef; fp--)
        fprintf(stderr, "\t%12.5e\n", *fp);

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
            fprintf(stderr, "sbisect: overflow min %f max %f\
							                         diff %e nroot %d n1 %d n2 %d\n",
                    min, max, max - min, nroot, n1, n2);
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
        fprintf(stderr, "sbisect: roots too close together\n");
        fprintf(stderr, "sbisect: overflow min %f max %f diff %e\
						                      nroot %d n1 %d n2 %d\n",
                min, max, max - min, nroot, n1, n2);
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

void fast_eigenvector_solver(double *eigv, int neig, Eigen::Matrix<double, 9, 9> &AM,
                             Eigen::Matrix<std::complex<double>, 3, 9> &sols) {
    static const int ind[] = {2, 4, 6, 9, 12, 15, 18, 22, 25, 29};
    // Truncated action matrix containing non-trivial rows
    Eigen::Matrix<double, 10, 9> AMs;
    double zi[5];

    for (int i = 0; i < 10; i++) {
        AMs.row(i) = AM.row(ind[i]);
    }
    for (int i = 0; i < neig; i++) {
        zi[0] = eigv[i];
        for (int j = 1; j < 5; j++) {
            zi[j] = zi[j - 1] * eigv[i];
        }
        Eigen::Matrix<double, 10, 10> AA;
        AA.col(0) = AMs.col(2);
        AA.col(1) = AMs.col(3) + zi[0] * AMs.col(4);
        AA.col(2) = AMs.col(5) + zi[0] * AMs.col(6);
        AA.col(3) = AMs.col(1) + zi[0] * AMs.col(7) + zi[1] * AMs.col(8) + zi[2] * AMs.col(9);
        AA.col(4) = AMs.col(11) + zi[0] * AMs.col(12);
        AA.col(5) = AMs.col(13) + zi[0] * AMs.col(14) + zi[1] * AMs.col(15);
        AA.col(6) = AMs.col(10) + zi[0] * AMs.col(16) + zi[1] * AMs.col(17) + zi[2] * AMs.col(18);
        AA.col(7) = AMs.col(20) + zi[0] * AMs.col(21) + zi[1] * AMs.col(22);
        AA.col(8) = AMs.col(19) + zi[0] * AMs.col(23) + zi[1] * AMs.col(24) + zi[2] * AMs.col(25);
        AA.col(9) = AMs.col(0) + zi[0] * AMs.col(26) + zi[1] * AMs.col(27) + zi[2] * AMs.col(28) + zi[3] * AMs.col(29);
        AA(0, 0) = AA(0, 0) - zi[0];
        AA(1, 1) = AA(1, 1) - zi[1];
        AA(2, 2) = AA(2, 2) - zi[1];
        AA(3, 3) = AA(3, 3) - zi[3];
        AA(4, 4) = AA(4, 4) - zi[1];
        AA(5, 5) = AA(5, 5) - zi[2];
        AA(6, 6) = AA(6, 6) - zi[3];
        AA(7, 7) = AA(7, 7) - zi[2];
        AA(8, 8) = AA(8, 8) - zi[3];
        AA(9, 9) = AA(9, 9) - zi[4];

        Eigen::Matrix<double, 9, 1> s = AA.leftCols(9).colPivHouseholderQr().solve(-AA.col(9));
        sols(0, i) = s(3);
        sols(1, i) = s(6);
        sols(2, i) = s(8);
        sols(3, i) = zi[0];
    }
}

int solver_relpose_6pt_onefocal(const Eigen::VectorXd &data, Eigen::Matrix<std::complex<double>, 3, 9> &sols) {
    // Compute coefficients
    const double *d = data.data();
    Eigen::VectorXd coeffs(190);
    coeffs[0] = d[9] * std::pow(d[15], 2) + 2 * d[10] * d[15] * d[16] - d[9] * std::pow(d[16], 2) +
                2 * d[11] * d[15] * d[17] - d[9] * std::pow(d[17], 2);
    coeffs[1] = std::pow(d[15], 2) * d[18] - std::pow(d[16], 2) * d[18] - std::pow(d[17], 2) * d[18] +
                2 * d[15] * d[16] * d[19] + 2 * d[15] * d[17] * d[20] + 2 * d[9] * d[15] * d[24] +
                2 * d[10] * d[16] * d[24] + 2 * d[11] * d[17] * d[24] + 2 * d[10] * d[15] * d[25] -
                2 * d[9] * d[16] * d[25] + 2 * d[11] * d[15] * d[26] - 2 * d[9] * d[17] * d[26];
    coeffs[2] = 2 * d[15] * d[18] * d[24] + 2 * d[16] * d[19] * d[24] + 2 * d[17] * d[20] * d[24] +
                d[9] * std::pow(d[24], 2) - 2 * d[16] * d[18] * d[25] + 2 * d[15] * d[19] * d[25] +
                2 * d[10] * d[24] * d[25] - d[9] * std::pow(d[25], 2) - 2 * d[17] * d[18] * d[26] +
                2 * d[15] * d[20] * d[26] + 2 * d[11] * d[24] * d[26] - d[9] * std::pow(d[26], 2);
    coeffs[3] = d[18] * std::pow(d[24], 2) + 2 * d[19] * d[24] * d[25] - d[18] * std::pow(d[25], 2) +
                2 * d[20] * d[24] * d[26] - d[18] * std::pow(d[26], 2);
    coeffs[4] = std::pow(d[9], 3) + d[9] * std::pow(d[10], 2) + d[9] * std::pow(d[11], 2) + d[9] * std::pow(d[12], 2) +
                2 * d[10] * d[12] * d[13] - d[9] * std::pow(d[13], 2) + 2 * d[11] * d[12] * d[14] -
                d[9] * std::pow(d[14], 2);
    coeffs[5] = 3 * std::pow(d[9], 2) * d[18] + std::pow(d[10], 2) * d[18] + std::pow(d[11], 2) * d[18] +
                std::pow(d[12], 2) * d[18] - std::pow(d[13], 2) * d[18] - std::pow(d[14], 2) * d[18] +
                2 * d[9] * d[10] * d[19] + 2 * d[12] * d[13] * d[19] + 2 * d[9] * d[11] * d[20] +
                2 * d[12] * d[14] * d[20] + 2 * d[9] * d[12] * d[21] + 2 * d[10] * d[13] * d[21] +
                2 * d[11] * d[14] * d[21] + 2 * d[10] * d[12] * d[22] - 2 * d[9] * d[13] * d[22] +
                2 * d[11] * d[12] * d[23] - 2 * d[9] * d[14] * d[23];
    coeffs[6] = 3 * d[9] * std::pow(d[18], 2) + 2 * d[10] * d[18] * d[19] + d[9] * std::pow(d[19], 2) +
                2 * d[11] * d[18] * d[20] + d[9] * std::pow(d[20], 2) + 2 * d[12] * d[18] * d[21] +
                2 * d[13] * d[19] * d[21] + 2 * d[14] * d[20] * d[21] + d[9] * std::pow(d[21], 2) -
                2 * d[13] * d[18] * d[22] + 2 * d[12] * d[19] * d[22] + 2 * d[10] * d[21] * d[22] -
                d[9] * std::pow(d[22], 2) - 2 * d[14] * d[18] * d[23] + 2 * d[12] * d[20] * d[23] +
                2 * d[11] * d[21] * d[23] - d[9] * std::pow(d[23], 2);
    coeffs[7] = std::pow(d[18], 3) + d[18] * std::pow(d[19], 2) + d[18] * std::pow(d[20], 2) +
                d[18] * std::pow(d[21], 2) + 2 * d[19] * d[21] * d[22] - d[18] * std::pow(d[22], 2) +
                2 * d[20] * d[21] * d[23] - d[18] * std::pow(d[23], 2);
    coeffs[8] = 2 * d[6] * d[9] * d[15] + 2 * d[7] * d[10] * d[15] + 2 * d[8] * d[11] * d[15] +
                d[0] * std::pow(d[15], 2) - 2 * d[7] * d[9] * d[16] + 2 * d[6] * d[10] * d[16] +
                2 * d[1] * d[15] * d[16] - d[0] * std::pow(d[16], 2) - 2 * d[8] * d[9] * d[17] +
                2 * d[6] * d[11] * d[17] + 2 * d[2] * d[15] * d[17] - d[0] * std::pow(d[17], 2);
    coeffs[9] = 2 * d[6] * d[15] * d[18] - 2 * d[7] * d[16] * d[18] - 2 * d[8] * d[17] * d[18] +
                2 * d[7] * d[15] * d[19] + 2 * d[6] * d[16] * d[19] + 2 * d[8] * d[15] * d[20] +
                2 * d[6] * d[17] * d[20] + 2 * d[6] * d[9] * d[24] + 2 * d[7] * d[10] * d[24] +
                2 * d[8] * d[11] * d[24] + 2 * d[0] * d[15] * d[24] + 2 * d[1] * d[16] * d[24] +
                2 * d[2] * d[17] * d[24] - 2 * d[7] * d[9] * d[25] + 2 * d[6] * d[10] * d[25] +
                2 * d[1] * d[15] * d[25] - 2 * d[0] * d[16] * d[25] - 2 * d[8] * d[9] * d[26] +
                2 * d[6] * d[11] * d[26] + 2 * d[2] * d[15] * d[26] - 2 * d[0] * d[17] * d[26];
    coeffs[10] = 2 * d[6] * d[18] * d[24] + 2 * d[7] * d[19] * d[24] + 2 * d[8] * d[20] * d[24] +
                 d[0] * std::pow(d[24], 2) - 2 * d[7] * d[18] * d[25] + 2 * d[6] * d[19] * d[25] +
                 2 * d[1] * d[24] * d[25] - d[0] * std::pow(d[25], 2) - 2 * d[8] * d[18] * d[26] +
                 2 * d[6] * d[20] * d[26] + 2 * d[2] * d[24] * d[26] - d[0] * std::pow(d[26], 2);
    coeffs[11] = 3 * d[0] * std::pow(d[9], 2) + 2 * d[1] * d[9] * d[10] + d[0] * std::pow(d[10], 2) +
                 2 * d[2] * d[9] * d[11] + d[0] * std::pow(d[11], 2) + 2 * d[3] * d[9] * d[12] +
                 2 * d[4] * d[10] * d[12] + 2 * d[5] * d[11] * d[12] + d[0] * std::pow(d[12], 2) -
                 2 * d[4] * d[9] * d[13] + 2 * d[3] * d[10] * d[13] + 2 * d[1] * d[12] * d[13] -
                 d[0] * std::pow(d[13], 2) - 2 * d[5] * d[9] * d[14] + 2 * d[3] * d[11] * d[14] +
                 2 * d[2] * d[12] * d[14] - d[0] * std::pow(d[14], 2);
    coeffs[12] =
        6 * d[0] * d[9] * d[18] + 2 * d[1] * d[10] * d[18] + 2 * d[2] * d[11] * d[18] + 2 * d[3] * d[12] * d[18] -
        2 * d[4] * d[13] * d[18] - 2 * d[5] * d[14] * d[18] + 2 * d[1] * d[9] * d[19] + 2 * d[0] * d[10] * d[19] +
        2 * d[4] * d[12] * d[19] + 2 * d[3] * d[13] * d[19] + 2 * d[2] * d[9] * d[20] + 2 * d[0] * d[11] * d[20] +
        2 * d[5] * d[12] * d[20] + 2 * d[3] * d[14] * d[20] + 2 * d[3] * d[9] * d[21] + 2 * d[4] * d[10] * d[21] +
        2 * d[5] * d[11] * d[21] + 2 * d[0] * d[12] * d[21] + 2 * d[1] * d[13] * d[21] + 2 * d[2] * d[14] * d[21] -
        2 * d[4] * d[9] * d[22] + 2 * d[3] * d[10] * d[22] + 2 * d[1] * d[12] * d[22] - 2 * d[0] * d[13] * d[22] -
        2 * d[5] * d[9] * d[23] + 2 * d[3] * d[11] * d[23] + 2 * d[2] * d[12] * d[23] - 2 * d[0] * d[14] * d[23];
    coeffs[13] = 3 * d[0] * std::pow(d[18], 2) + 2 * d[1] * d[18] * d[19] + d[0] * std::pow(d[19], 2) +
                 2 * d[2] * d[18] * d[20] + d[0] * std::pow(d[20], 2) + 2 * d[3] * d[18] * d[21] +
                 2 * d[4] * d[19] * d[21] + 2 * d[5] * d[20] * d[21] + d[0] * std::pow(d[21], 2) -
                 2 * d[4] * d[18] * d[22] + 2 * d[3] * d[19] * d[22] + 2 * d[1] * d[21] * d[22] -
                 d[0] * std::pow(d[22], 2) - 2 * d[5] * d[18] * d[23] + 2 * d[3] * d[20] * d[23] +
                 2 * d[2] * d[21] * d[23] - d[0] * std::pow(d[23], 2);
    coeffs[14] = std::pow(d[6], 2) * d[9] - std::pow(d[7], 2) * d[9] - std::pow(d[8], 2) * d[9] +
                 2 * d[6] * d[7] * d[10] + 2 * d[6] * d[8] * d[11] + 2 * d[0] * d[6] * d[15] + 2 * d[1] * d[7] * d[15] +
                 2 * d[2] * d[8] * d[15] + 2 * d[1] * d[6] * d[16] - 2 * d[0] * d[7] * d[16] + 2 * d[2] * d[6] * d[17] -
                 2 * d[0] * d[8] * d[17];
    coeffs[15] = std::pow(d[6], 2) * d[18] - std::pow(d[7], 2) * d[18] - std::pow(d[8], 2) * d[18] +
                 2 * d[6] * d[7] * d[19] + 2 * d[6] * d[8] * d[20] + 2 * d[0] * d[6] * d[24] + 2 * d[1] * d[7] * d[24] +
                 2 * d[2] * d[8] * d[24] + 2 * d[1] * d[6] * d[25] - 2 * d[0] * d[7] * d[25] + 2 * d[2] * d[6] * d[26] -
                 2 * d[0] * d[8] * d[26];
    coeffs[16] = 3 * std::pow(d[0], 2) * d[9] + std::pow(d[1], 2) * d[9] + std::pow(d[2], 2) * d[9] +
                 std::pow(d[3], 2) * d[9] - std::pow(d[4], 2) * d[9] - std::pow(d[5], 2) * d[9] +
                 2 * d[0] * d[1] * d[10] + 2 * d[3] * d[4] * d[10] + 2 * d[0] * d[2] * d[11] + 2 * d[3] * d[5] * d[11] +
                 2 * d[0] * d[3] * d[12] + 2 * d[1] * d[4] * d[12] + 2 * d[2] * d[5] * d[12] + 2 * d[1] * d[3] * d[13] -
                 2 * d[0] * d[4] * d[13] + 2 * d[2] * d[3] * d[14] - 2 * d[0] * d[5] * d[14];
    coeffs[17] = 3 * std::pow(d[0], 2) * d[18] + std::pow(d[1], 2) * d[18] + std::pow(d[2], 2) * d[18] +
                 std::pow(d[3], 2) * d[18] - std::pow(d[4], 2) * d[18] - std::pow(d[5], 2) * d[18] +
                 2 * d[0] * d[1] * d[19] + 2 * d[3] * d[4] * d[19] + 2 * d[0] * d[2] * d[20] + 2 * d[3] * d[5] * d[20] +
                 2 * d[0] * d[3] * d[21] + 2 * d[1] * d[4] * d[21] + 2 * d[2] * d[5] * d[21] + 2 * d[1] * d[3] * d[22] -
                 2 * d[0] * d[4] * d[22] + 2 * d[2] * d[3] * d[23] - 2 * d[0] * d[5] * d[23];
    coeffs[18] = d[0] * std::pow(d[6], 2) + 2 * d[1] * d[6] * d[7] - d[0] * std::pow(d[7], 2) + 2 * d[2] * d[6] * d[8] -
                 d[0] * std::pow(d[8], 2);
    coeffs[19] = std::pow(d[0], 3) + d[0] * std::pow(d[1], 2) + d[0] * std::pow(d[2], 2) + d[0] * std::pow(d[3], 2) +
                 2 * d[1] * d[3] * d[4] - d[0] * std::pow(d[4], 2) + 2 * d[2] * d[3] * d[5] - d[0] * std::pow(d[5], 2);
    coeffs[20] = -d[10] * std::pow(d[15], 2) + 2 * d[9] * d[15] * d[16] + d[10] * std::pow(d[16], 2) +
                 2 * d[11] * d[16] * d[17] - d[10] * std::pow(d[17], 2);
    coeffs[21] = 2 * d[15] * d[16] * d[18] - std::pow(d[15], 2) * d[19] + std::pow(d[16], 2) * d[19] -
                 std::pow(d[17], 2) * d[19] + 2 * d[16] * d[17] * d[20] - 2 * d[10] * d[15] * d[24] +
                 2 * d[9] * d[16] * d[24] + 2 * d[9] * d[15] * d[25] + 2 * d[10] * d[16] * d[25] +
                 2 * d[11] * d[17] * d[25] + 2 * d[11] * d[16] * d[26] - 2 * d[10] * d[17] * d[26];
    coeffs[22] = 2 * d[16] * d[18] * d[24] - 2 * d[15] * d[19] * d[24] - d[10] * std::pow(d[24], 2) +
                 2 * d[15] * d[18] * d[25] + 2 * d[16] * d[19] * d[25] + 2 * d[17] * d[20] * d[25] +
                 2 * d[9] * d[24] * d[25] + d[10] * std::pow(d[25], 2) - 2 * d[17] * d[19] * d[26] +
                 2 * d[16] * d[20] * d[26] + 2 * d[11] * d[25] * d[26] - d[10] * std::pow(d[26], 2);
    coeffs[23] = -d[19] * std::pow(d[24], 2) + 2 * d[18] * d[24] * d[25] + d[19] * std::pow(d[25], 2) +
                 2 * d[20] * d[25] * d[26] - d[19] * std::pow(d[26], 2);
    coeffs[24] = std::pow(d[9], 2) * d[10] + std::pow(d[10], 3) + d[10] * std::pow(d[11], 2) -
                 d[10] * std::pow(d[12], 2) + 2 * d[9] * d[12] * d[13] + d[10] * std::pow(d[13], 2) +
                 2 * d[11] * d[13] * d[14] - d[10] * std::pow(d[14], 2);
    coeffs[25] = 2 * d[9] * d[10] * d[18] + 2 * d[12] * d[13] * d[18] + std::pow(d[9], 2) * d[19] +
                 3 * std::pow(d[10], 2) * d[19] + std::pow(d[11], 2) * d[19] - std::pow(d[12], 2) * d[19] +
                 std::pow(d[13], 2) * d[19] - std::pow(d[14], 2) * d[19] + 2 * d[10] * d[11] * d[20] +
                 2 * d[13] * d[14] * d[20] - 2 * d[10] * d[12] * d[21] + 2 * d[9] * d[13] * d[21] +
                 2 * d[9] * d[12] * d[22] + 2 * d[10] * d[13] * d[22] + 2 * d[11] * d[14] * d[22] +
                 2 * d[11] * d[13] * d[23] - 2 * d[10] * d[14] * d[23];
    coeffs[26] = d[10] * std::pow(d[18], 2) + 2 * d[9] * d[18] * d[19] + 3 * d[10] * std::pow(d[19], 2) +
                 2 * d[11] * d[19] * d[20] + d[10] * std::pow(d[20], 2) + 2 * d[13] * d[18] * d[21] -
                 2 * d[12] * d[19] * d[21] - d[10] * std::pow(d[21], 2) + 2 * d[12] * d[18] * d[22] +
                 2 * d[13] * d[19] * d[22] + 2 * d[14] * d[20] * d[22] + 2 * d[9] * d[21] * d[22] +
                 d[10] * std::pow(d[22], 2) - 2 * d[14] * d[19] * d[23] + 2 * d[13] * d[20] * d[23] +
                 2 * d[11] * d[22] * d[23] - d[10] * std::pow(d[23], 2);
    coeffs[27] = std::pow(d[18], 2) * d[19] + std::pow(d[19], 3) + d[19] * std::pow(d[20], 2) -
                 d[19] * std::pow(d[21], 2) + 2 * d[18] * d[21] * d[22] + d[19] * std::pow(d[22], 2) +
                 2 * d[20] * d[22] * d[23] - d[19] * std::pow(d[23], 2);
    coeffs[28] = 2 * d[7] * d[9] * d[15] - 2 * d[6] * d[10] * d[15] - d[1] * std::pow(d[15], 2) +
                 2 * d[6] * d[9] * d[16] + 2 * d[7] * d[10] * d[16] + 2 * d[8] * d[11] * d[16] +
                 2 * d[0] * d[15] * d[16] + d[1] * std::pow(d[16], 2) - 2 * d[8] * d[10] * d[17] +
                 2 * d[7] * d[11] * d[17] + 2 * d[2] * d[16] * d[17] - d[1] * std::pow(d[17], 2);
    coeffs[29] = 2 * d[7] * d[15] * d[18] + 2 * d[6] * d[16] * d[18] - 2 * d[6] * d[15] * d[19] +
                 2 * d[7] * d[16] * d[19] - 2 * d[8] * d[17] * d[19] + 2 * d[8] * d[16] * d[20] +
                 2 * d[7] * d[17] * d[20] + 2 * d[7] * d[9] * d[24] - 2 * d[6] * d[10] * d[24] -
                 2 * d[1] * d[15] * d[24] + 2 * d[0] * d[16] * d[24] + 2 * d[6] * d[9] * d[25] +
                 2 * d[7] * d[10] * d[25] + 2 * d[8] * d[11] * d[25] + 2 * d[0] * d[15] * d[25] +
                 2 * d[1] * d[16] * d[25] + 2 * d[2] * d[17] * d[25] - 2 * d[8] * d[10] * d[26] +
                 2 * d[7] * d[11] * d[26] + 2 * d[2] * d[16] * d[26] - 2 * d[1] * d[17] * d[26];
    coeffs[30] = 2 * d[7] * d[18] * d[24] - 2 * d[6] * d[19] * d[24] - d[1] * std::pow(d[24], 2) +
                 2 * d[6] * d[18] * d[25] + 2 * d[7] * d[19] * d[25] + 2 * d[8] * d[20] * d[25] +
                 2 * d[0] * d[24] * d[25] + d[1] * std::pow(d[25], 2) - 2 * d[8] * d[19] * d[26] +
                 2 * d[7] * d[20] * d[26] + 2 * d[2] * d[25] * d[26] - d[1] * std::pow(d[26], 2);
    coeffs[31] = d[1] * std::pow(d[9], 2) + 2 * d[0] * d[9] * d[10] + 3 * d[1] * std::pow(d[10], 2) +
                 2 * d[2] * d[10] * d[11] + d[1] * std::pow(d[11], 2) + 2 * d[4] * d[9] * d[12] -
                 2 * d[3] * d[10] * d[12] - d[1] * std::pow(d[12], 2) + 2 * d[3] * d[9] * d[13] +
                 2 * d[4] * d[10] * d[13] + 2 * d[5] * d[11] * d[13] + 2 * d[0] * d[12] * d[13] +
                 d[1] * std::pow(d[13], 2) - 2 * d[5] * d[10] * d[14] + 2 * d[4] * d[11] * d[14] +
                 2 * d[2] * d[13] * d[14] - d[1] * std::pow(d[14], 2);
    coeffs[32] =
        2 * d[1] * d[9] * d[18] + 2 * d[0] * d[10] * d[18] + 2 * d[4] * d[12] * d[18] + 2 * d[3] * d[13] * d[18] +
        2 * d[0] * d[9] * d[19] + 6 * d[1] * d[10] * d[19] + 2 * d[2] * d[11] * d[19] - 2 * d[3] * d[12] * d[19] +
        2 * d[4] * d[13] * d[19] - 2 * d[5] * d[14] * d[19] + 2 * d[2] * d[10] * d[20] + 2 * d[1] * d[11] * d[20] +
        2 * d[5] * d[13] * d[20] + 2 * d[4] * d[14] * d[20] + 2 * d[4] * d[9] * d[21] - 2 * d[3] * d[10] * d[21] -
        2 * d[1] * d[12] * d[21] + 2 * d[0] * d[13] * d[21] + 2 * d[3] * d[9] * d[22] + 2 * d[4] * d[10] * d[22] +
        2 * d[5] * d[11] * d[22] + 2 * d[0] * d[12] * d[22] + 2 * d[1] * d[13] * d[22] + 2 * d[2] * d[14] * d[22] -
        2 * d[5] * d[10] * d[23] + 2 * d[4] * d[11] * d[23] + 2 * d[2] * d[13] * d[23] - 2 * d[1] * d[14] * d[23];
    coeffs[33] = d[1] * std::pow(d[18], 2) + 2 * d[0] * d[18] * d[19] + 3 * d[1] * std::pow(d[19], 2) +
                 2 * d[2] * d[19] * d[20] + d[1] * std::pow(d[20], 2) + 2 * d[4] * d[18] * d[21] -
                 2 * d[3] * d[19] * d[21] - d[1] * std::pow(d[21], 2) + 2 * d[3] * d[18] * d[22] +
                 2 * d[4] * d[19] * d[22] + 2 * d[5] * d[20] * d[22] + 2 * d[0] * d[21] * d[22] +
                 d[1] * std::pow(d[22], 2) - 2 * d[5] * d[19] * d[23] + 2 * d[4] * d[20] * d[23] +
                 2 * d[2] * d[22] * d[23] - d[1] * std::pow(d[23], 2);
    coeffs[34] = 2 * d[6] * d[7] * d[9] - std::pow(d[6], 2) * d[10] + std::pow(d[7], 2) * d[10] -
                 std::pow(d[8], 2) * d[10] + 2 * d[7] * d[8] * d[11] - 2 * d[1] * d[6] * d[15] +
                 2 * d[0] * d[7] * d[15] + 2 * d[0] * d[6] * d[16] + 2 * d[1] * d[7] * d[16] + 2 * d[2] * d[8] * d[16] +
                 2 * d[2] * d[7] * d[17] - 2 * d[1] * d[8] * d[17];
    coeffs[35] = 2 * d[6] * d[7] * d[18] - std::pow(d[6], 2) * d[19] + std::pow(d[7], 2) * d[19] -
                 std::pow(d[8], 2) * d[19] + 2 * d[7] * d[8] * d[20] - 2 * d[1] * d[6] * d[24] +
                 2 * d[0] * d[7] * d[24] + 2 * d[0] * d[6] * d[25] + 2 * d[1] * d[7] * d[25] + 2 * d[2] * d[8] * d[25] +
                 2 * d[2] * d[7] * d[26] - 2 * d[1] * d[8] * d[26];
    coeffs[36] = 2 * d[0] * d[1] * d[9] + 2 * d[3] * d[4] * d[9] + std::pow(d[0], 2) * d[10] +
                 3 * std::pow(d[1], 2) * d[10] + std::pow(d[2], 2) * d[10] - std::pow(d[3], 2) * d[10] +
                 std::pow(d[4], 2) * d[10] - std::pow(d[5], 2) * d[10] + 2 * d[1] * d[2] * d[11] +
                 2 * d[4] * d[5] * d[11] - 2 * d[1] * d[3] * d[12] + 2 * d[0] * d[4] * d[12] + 2 * d[0] * d[3] * d[13] +
                 2 * d[1] * d[4] * d[13] + 2 * d[2] * d[5] * d[13] + 2 * d[2] * d[4] * d[14] - 2 * d[1] * d[5] * d[14];
    coeffs[37] = 2 * d[0] * d[1] * d[18] + 2 * d[3] * d[4] * d[18] + std::pow(d[0], 2) * d[19] +
                 3 * std::pow(d[1], 2) * d[19] + std::pow(d[2], 2) * d[19] - std::pow(d[3], 2) * d[19] +
                 std::pow(d[4], 2) * d[19] - std::pow(d[5], 2) * d[19] + 2 * d[1] * d[2] * d[20] +
                 2 * d[4] * d[5] * d[20] - 2 * d[1] * d[3] * d[21] + 2 * d[0] * d[4] * d[21] + 2 * d[0] * d[3] * d[22] +
                 2 * d[1] * d[4] * d[22] + 2 * d[2] * d[5] * d[22] + 2 * d[2] * d[4] * d[23] - 2 * d[1] * d[5] * d[23];
    coeffs[38] = -d[1] * std::pow(d[6], 2) + 2 * d[0] * d[6] * d[7] + d[1] * std::pow(d[7], 2) +
                 2 * d[2] * d[7] * d[8] - d[1] * std::pow(d[8], 2);
    coeffs[39] = std::pow(d[0], 2) * d[1] + std::pow(d[1], 3) + d[1] * std::pow(d[2], 2) - d[1] * std::pow(d[3], 2) +
                 2 * d[0] * d[3] * d[4] + d[1] * std::pow(d[4], 2) + 2 * d[2] * d[4] * d[5] - d[1] * std::pow(d[5], 2);
    coeffs[40] = -d[11] * std::pow(d[15], 2) - d[11] * std::pow(d[16], 2) + 2 * d[9] * d[15] * d[17] +
                 2 * d[10] * d[16] * d[17] + d[11] * std::pow(d[17], 2);
    coeffs[41] = 2 * d[15] * d[17] * d[18] + 2 * d[16] * d[17] * d[19] - std::pow(d[15], 2) * d[20] -
                 std::pow(d[16], 2) * d[20] + std::pow(d[17], 2) * d[20] - 2 * d[11] * d[15] * d[24] +
                 2 * d[9] * d[17] * d[24] - 2 * d[11] * d[16] * d[25] + 2 * d[10] * d[17] * d[25] +
                 2 * d[9] * d[15] * d[26] + 2 * d[10] * d[16] * d[26] + 2 * d[11] * d[17] * d[26];
    coeffs[42] = 2 * d[17] * d[18] * d[24] - 2 * d[15] * d[20] * d[24] - d[11] * std::pow(d[24], 2) +
                 2 * d[17] * d[19] * d[25] - 2 * d[16] * d[20] * d[25] - d[11] * std::pow(d[25], 2) +
                 2 * d[15] * d[18] * d[26] + 2 * d[16] * d[19] * d[26] + 2 * d[17] * d[20] * d[26] +
                 2 * d[9] * d[24] * d[26] + 2 * d[10] * d[25] * d[26] + d[11] * std::pow(d[26], 2);
    coeffs[43] = -d[20] * std::pow(d[24], 2) - d[20] * std::pow(d[25], 2) + 2 * d[18] * d[24] * d[26] +
                 2 * d[19] * d[25] * d[26] + d[20] * std::pow(d[26], 2);
    coeffs[44] = std::pow(d[9], 2) * d[11] + std::pow(d[10], 2) * d[11] + std::pow(d[11], 3) -
                 d[11] * std::pow(d[12], 2) - d[11] * std::pow(d[13], 2) + 2 * d[9] * d[12] * d[14] +
                 2 * d[10] * d[13] * d[14] + d[11] * std::pow(d[14], 2);
    coeffs[45] = 2 * d[9] * d[11] * d[18] + 2 * d[12] * d[14] * d[18] + 2 * d[10] * d[11] * d[19] +
                 2 * d[13] * d[14] * d[19] + std::pow(d[9], 2) * d[20] + std::pow(d[10], 2) * d[20] +
                 3 * std::pow(d[11], 2) * d[20] - std::pow(d[12], 2) * d[20] - std::pow(d[13], 2) * d[20] +
                 std::pow(d[14], 2) * d[20] - 2 * d[11] * d[12] * d[21] + 2 * d[9] * d[14] * d[21] -
                 2 * d[11] * d[13] * d[22] + 2 * d[10] * d[14] * d[22] + 2 * d[9] * d[12] * d[23] +
                 2 * d[10] * d[13] * d[23] + 2 * d[11] * d[14] * d[23];
    coeffs[46] = d[11] * std::pow(d[18], 2) + d[11] * std::pow(d[19], 2) + 2 * d[9] * d[18] * d[20] +
                 2 * d[10] * d[19] * d[20] + 3 * d[11] * std::pow(d[20], 2) + 2 * d[14] * d[18] * d[21] -
                 2 * d[12] * d[20] * d[21] - d[11] * std::pow(d[21], 2) + 2 * d[14] * d[19] * d[22] -
                 2 * d[13] * d[20] * d[22] - d[11] * std::pow(d[22], 2) + 2 * d[12] * d[18] * d[23] +
                 2 * d[13] * d[19] * d[23] + 2 * d[14] * d[20] * d[23] + 2 * d[9] * d[21] * d[23] +
                 2 * d[10] * d[22] * d[23] + d[11] * std::pow(d[23], 2);
    coeffs[47] = std::pow(d[18], 2) * d[20] + std::pow(d[19], 2) * d[20] + std::pow(d[20], 3) -
                 d[20] * std::pow(d[21], 2) - d[20] * std::pow(d[22], 2) + 2 * d[18] * d[21] * d[23] +
                 2 * d[19] * d[22] * d[23] + d[20] * std::pow(d[23], 2);
    coeffs[48] = 2 * d[8] * d[9] * d[15] - 2 * d[6] * d[11] * d[15] - d[2] * std::pow(d[15], 2) +
                 2 * d[8] * d[10] * d[16] - 2 * d[7] * d[11] * d[16] - d[2] * std::pow(d[16], 2) +
                 2 * d[6] * d[9] * d[17] + 2 * d[7] * d[10] * d[17] + 2 * d[8] * d[11] * d[17] +
                 2 * d[0] * d[15] * d[17] + 2 * d[1] * d[16] * d[17] + d[2] * std::pow(d[17], 2);
    coeffs[49] = 2 * d[8] * d[15] * d[18] + 2 * d[6] * d[17] * d[18] + 2 * d[8] * d[16] * d[19] +
                 2 * d[7] * d[17] * d[19] - 2 * d[6] * d[15] * d[20] - 2 * d[7] * d[16] * d[20] +
                 2 * d[8] * d[17] * d[20] + 2 * d[8] * d[9] * d[24] - 2 * d[6] * d[11] * d[24] -
                 2 * d[2] * d[15] * d[24] + 2 * d[0] * d[17] * d[24] + 2 * d[8] * d[10] * d[25] -
                 2 * d[7] * d[11] * d[25] - 2 * d[2] * d[16] * d[25] + 2 * d[1] * d[17] * d[25] +
                 2 * d[6] * d[9] * d[26] + 2 * d[7] * d[10] * d[26] + 2 * d[8] * d[11] * d[26] +
                 2 * d[0] * d[15] * d[26] + 2 * d[1] * d[16] * d[26] + 2 * d[2] * d[17] * d[26];
    coeffs[50] = 2 * d[8] * d[18] * d[24] - 2 * d[6] * d[20] * d[24] - d[2] * std::pow(d[24], 2) +
                 2 * d[8] * d[19] * d[25] - 2 * d[7] * d[20] * d[25] - d[2] * std::pow(d[25], 2) +
                 2 * d[6] * d[18] * d[26] + 2 * d[7] * d[19] * d[26] + 2 * d[8] * d[20] * d[26] +
                 2 * d[0] * d[24] * d[26] + 2 * d[1] * d[25] * d[26] + d[2] * std::pow(d[26], 2);
    coeffs[51] = d[2] * std::pow(d[9], 2) + d[2] * std::pow(d[10], 2) + 2 * d[0] * d[9] * d[11] +
                 2 * d[1] * d[10] * d[11] + 3 * d[2] * std::pow(d[11], 2) + 2 * d[5] * d[9] * d[12] -
                 2 * d[3] * d[11] * d[12] - d[2] * std::pow(d[12], 2) + 2 * d[5] * d[10] * d[13] -
                 2 * d[4] * d[11] * d[13] - d[2] * std::pow(d[13], 2) + 2 * d[3] * d[9] * d[14] +
                 2 * d[4] * d[10] * d[14] + 2 * d[5] * d[11] * d[14] + 2 * d[0] * d[12] * d[14] +
                 2 * d[1] * d[13] * d[14] + d[2] * std::pow(d[14], 2);
    coeffs[52] =
        2 * d[2] * d[9] * d[18] + 2 * d[0] * d[11] * d[18] + 2 * d[5] * d[12] * d[18] + 2 * d[3] * d[14] * d[18] +
        2 * d[2] * d[10] * d[19] + 2 * d[1] * d[11] * d[19] + 2 * d[5] * d[13] * d[19] + 2 * d[4] * d[14] * d[19] +
        2 * d[0] * d[9] * d[20] + 2 * d[1] * d[10] * d[20] + 6 * d[2] * d[11] * d[20] - 2 * d[3] * d[12] * d[20] -
        2 * d[4] * d[13] * d[20] + 2 * d[5] * d[14] * d[20] + 2 * d[5] * d[9] * d[21] - 2 * d[3] * d[11] * d[21] -
        2 * d[2] * d[12] * d[21] + 2 * d[0] * d[14] * d[21] + 2 * d[5] * d[10] * d[22] - 2 * d[4] * d[11] * d[22] -
        2 * d[2] * d[13] * d[22] + 2 * d[1] * d[14] * d[22] + 2 * d[3] * d[9] * d[23] + 2 * d[4] * d[10] * d[23] +
        2 * d[5] * d[11] * d[23] + 2 * d[0] * d[12] * d[23] + 2 * d[1] * d[13] * d[23] + 2 * d[2] * d[14] * d[23];
    coeffs[53] = d[2] * std::pow(d[18], 2) + d[2] * std::pow(d[19], 2) + 2 * d[0] * d[18] * d[20] +
                 2 * d[1] * d[19] * d[20] + 3 * d[2] * std::pow(d[20], 2) + 2 * d[5] * d[18] * d[21] -
                 2 * d[3] * d[20] * d[21] - d[2] * std::pow(d[21], 2) + 2 * d[5] * d[19] * d[22] -
                 2 * d[4] * d[20] * d[22] - d[2] * std::pow(d[22], 2) + 2 * d[3] * d[18] * d[23] +
                 2 * d[4] * d[19] * d[23] + 2 * d[5] * d[20] * d[23] + 2 * d[0] * d[21] * d[23] +
                 2 * d[1] * d[22] * d[23] + d[2] * std::pow(d[23], 2);
    coeffs[54] = 2 * d[6] * d[8] * d[9] + 2 * d[7] * d[8] * d[10] - std::pow(d[6], 2) * d[11] -
                 std::pow(d[7], 2) * d[11] + std::pow(d[8], 2) * d[11] - 2 * d[2] * d[6] * d[15] +
                 2 * d[0] * d[8] * d[15] - 2 * d[2] * d[7] * d[16] + 2 * d[1] * d[8] * d[16] + 2 * d[0] * d[6] * d[17] +
                 2 * d[1] * d[7] * d[17] + 2 * d[2] * d[8] * d[17];
    coeffs[55] = 2 * d[6] * d[8] * d[18] + 2 * d[7] * d[8] * d[19] - std::pow(d[6], 2) * d[20] -
                 std::pow(d[7], 2) * d[20] + std::pow(d[8], 2) * d[20] - 2 * d[2] * d[6] * d[24] +
                 2 * d[0] * d[8] * d[24] - 2 * d[2] * d[7] * d[25] + 2 * d[1] * d[8] * d[25] + 2 * d[0] * d[6] * d[26] +
                 2 * d[1] * d[7] * d[26] + 2 * d[2] * d[8] * d[26];
    coeffs[56] = 2 * d[0] * d[2] * d[9] + 2 * d[3] * d[5] * d[9] + 2 * d[1] * d[2] * d[10] + 2 * d[4] * d[5] * d[10] +
                 std::pow(d[0], 2) * d[11] + std::pow(d[1], 2) * d[11] + 3 * std::pow(d[2], 2) * d[11] -
                 std::pow(d[3], 2) * d[11] - std::pow(d[4], 2) * d[11] + std::pow(d[5], 2) * d[11] -
                 2 * d[2] * d[3] * d[12] + 2 * d[0] * d[5] * d[12] - 2 * d[2] * d[4] * d[13] + 2 * d[1] * d[5] * d[13] +
                 2 * d[0] * d[3] * d[14] + 2 * d[1] * d[4] * d[14] + 2 * d[2] * d[5] * d[14];
    coeffs[57] = 2 * d[0] * d[2] * d[18] + 2 * d[3] * d[5] * d[18] + 2 * d[1] * d[2] * d[19] + 2 * d[4] * d[5] * d[19] +
                 std::pow(d[0], 2) * d[20] + std::pow(d[1], 2) * d[20] + 3 * std::pow(d[2], 2) * d[20] -
                 std::pow(d[3], 2) * d[20] - std::pow(d[4], 2) * d[20] + std::pow(d[5], 2) * d[20] -
                 2 * d[2] * d[3] * d[21] + 2 * d[0] * d[5] * d[21] - 2 * d[2] * d[4] * d[22] + 2 * d[1] * d[5] * d[22] +
                 2 * d[0] * d[3] * d[23] + 2 * d[1] * d[4] * d[23] + 2 * d[2] * d[5] * d[23];
    coeffs[58] = -d[2] * std::pow(d[6], 2) - d[2] * std::pow(d[7], 2) + 2 * d[0] * d[6] * d[8] +
                 2 * d[1] * d[7] * d[8] + d[2] * std::pow(d[8], 2);
    coeffs[59] = std::pow(d[0], 2) * d[2] + std::pow(d[1], 2) * d[2] + std::pow(d[2], 3) - d[2] * std::pow(d[3], 2) -
                 d[2] * std::pow(d[4], 2) + 2 * d[0] * d[3] * d[5] + 2 * d[1] * d[4] * d[5] + d[2] * std::pow(d[5], 2);
    coeffs[60] = d[12] * std::pow(d[15], 2) + 2 * d[13] * d[15] * d[16] - d[12] * std::pow(d[16], 2) +
                 2 * d[14] * d[15] * d[17] - d[12] * std::pow(d[17], 2);
    coeffs[61] = std::pow(d[15], 2) * d[21] - std::pow(d[16], 2) * d[21] - std::pow(d[17], 2) * d[21] +
                 2 * d[15] * d[16] * d[22] + 2 * d[15] * d[17] * d[23] + 2 * d[12] * d[15] * d[24] +
                 2 * d[13] * d[16] * d[24] + 2 * d[14] * d[17] * d[24] + 2 * d[13] * d[15] * d[25] -
                 2 * d[12] * d[16] * d[25] + 2 * d[14] * d[15] * d[26] - 2 * d[12] * d[17] * d[26];
    coeffs[62] = 2 * d[15] * d[21] * d[24] + 2 * d[16] * d[22] * d[24] + 2 * d[17] * d[23] * d[24] +
                 d[12] * std::pow(d[24], 2) - 2 * d[16] * d[21] * d[25] + 2 * d[15] * d[22] * d[25] +
                 2 * d[13] * d[24] * d[25] - d[12] * std::pow(d[25], 2) - 2 * d[17] * d[21] * d[26] +
                 2 * d[15] * d[23] * d[26] + 2 * d[14] * d[24] * d[26] - d[12] * std::pow(d[26], 2);
    coeffs[63] = d[21] * std::pow(d[24], 2) + 2 * d[22] * d[24] * d[25] - d[21] * std::pow(d[25], 2) +
                 2 * d[23] * d[24] * d[26] - d[21] * std::pow(d[26], 2);
    coeffs[64] = std::pow(d[9], 2) * d[12] - std::pow(d[10], 2) * d[12] - std::pow(d[11], 2) * d[12] +
                 std::pow(d[12], 3) + 2 * d[9] * d[10] * d[13] + d[12] * std::pow(d[13], 2) + 2 * d[9] * d[11] * d[14] +
                 d[12] * std::pow(d[14], 2);
    coeffs[65] = 2 * d[9] * d[12] * d[18] + 2 * d[10] * d[13] * d[18] + 2 * d[11] * d[14] * d[18] -
                 2 * d[10] * d[12] * d[19] + 2 * d[9] * d[13] * d[19] - 2 * d[11] * d[12] * d[20] +
                 2 * d[9] * d[14] * d[20] + std::pow(d[9], 2) * d[21] - std::pow(d[10], 2) * d[21] -
                 std::pow(d[11], 2) * d[21] + 3 * std::pow(d[12], 2) * d[21] + std::pow(d[13], 2) * d[21] +
                 std::pow(d[14], 2) * d[21] + 2 * d[9] * d[10] * d[22] + 2 * d[12] * d[13] * d[22] +
                 2 * d[9] * d[11] * d[23] + 2 * d[12] * d[14] * d[23];
    coeffs[66] = d[12] * std::pow(d[18], 2) + 2 * d[13] * d[18] * d[19] - d[12] * std::pow(d[19], 2) +
                 2 * d[14] * d[18] * d[20] - d[12] * std::pow(d[20], 2) + 2 * d[9] * d[18] * d[21] -
                 2 * d[10] * d[19] * d[21] - 2 * d[11] * d[20] * d[21] + 3 * d[12] * std::pow(d[21], 2) +
                 2 * d[10] * d[18] * d[22] + 2 * d[9] * d[19] * d[22] + 2 * d[13] * d[21] * d[22] +
                 d[12] * std::pow(d[22], 2) + 2 * d[11] * d[18] * d[23] + 2 * d[9] * d[20] * d[23] +
                 2 * d[14] * d[21] * d[23] + d[12] * std::pow(d[23], 2);
    coeffs[67] = std::pow(d[18], 2) * d[21] - std::pow(d[19], 2) * d[21] - std::pow(d[20], 2) * d[21] +
                 std::pow(d[21], 3) + 2 * d[18] * d[19] * d[22] + d[21] * std::pow(d[22], 2) +
                 2 * d[18] * d[20] * d[23] + d[21] * std::pow(d[23], 2);
    coeffs[68] = 2 * d[6] * d[12] * d[15] + 2 * d[7] * d[13] * d[15] + 2 * d[8] * d[14] * d[15] +
                 d[3] * std::pow(d[15], 2) - 2 * d[7] * d[12] * d[16] + 2 * d[6] * d[13] * d[16] +
                 2 * d[4] * d[15] * d[16] - d[3] * std::pow(d[16], 2) - 2 * d[8] * d[12] * d[17] +
                 2 * d[6] * d[14] * d[17] + 2 * d[5] * d[15] * d[17] - d[3] * std::pow(d[17], 2);
    coeffs[69] = 2 * d[6] * d[15] * d[21] - 2 * d[7] * d[16] * d[21] - 2 * d[8] * d[17] * d[21] +
                 2 * d[7] * d[15] * d[22] + 2 * d[6] * d[16] * d[22] + 2 * d[8] * d[15] * d[23] +
                 2 * d[6] * d[17] * d[23] + 2 * d[6] * d[12] * d[24] + 2 * d[7] * d[13] * d[24] +
                 2 * d[8] * d[14] * d[24] + 2 * d[3] * d[15] * d[24] + 2 * d[4] * d[16] * d[24] +
                 2 * d[5] * d[17] * d[24] - 2 * d[7] * d[12] * d[25] + 2 * d[6] * d[13] * d[25] +
                 2 * d[4] * d[15] * d[25] - 2 * d[3] * d[16] * d[25] - 2 * d[8] * d[12] * d[26] +
                 2 * d[6] * d[14] * d[26] + 2 * d[5] * d[15] * d[26] - 2 * d[3] * d[17] * d[26];
    coeffs[70] = 2 * d[6] * d[21] * d[24] + 2 * d[7] * d[22] * d[24] + 2 * d[8] * d[23] * d[24] +
                 d[3] * std::pow(d[24], 2) - 2 * d[7] * d[21] * d[25] + 2 * d[6] * d[22] * d[25] +
                 2 * d[4] * d[24] * d[25] - d[3] * std::pow(d[25], 2) - 2 * d[8] * d[21] * d[26] +
                 2 * d[6] * d[23] * d[26] + 2 * d[5] * d[24] * d[26] - d[3] * std::pow(d[26], 2);
    coeffs[71] = d[3] * std::pow(d[9], 2) + 2 * d[4] * d[9] * d[10] - d[3] * std::pow(d[10], 2) +
                 2 * d[5] * d[9] * d[11] - d[3] * std::pow(d[11], 2) + 2 * d[0] * d[9] * d[12] -
                 2 * d[1] * d[10] * d[12] - 2 * d[2] * d[11] * d[12] + 3 * d[3] * std::pow(d[12], 2) +
                 2 * d[1] * d[9] * d[13] + 2 * d[0] * d[10] * d[13] + 2 * d[4] * d[12] * d[13] +
                 d[3] * std::pow(d[13], 2) + 2 * d[2] * d[9] * d[14] + 2 * d[0] * d[11] * d[14] +
                 2 * d[5] * d[12] * d[14] + d[3] * std::pow(d[14], 2);
    coeffs[72] =
        2 * d[3] * d[9] * d[18] + 2 * d[4] * d[10] * d[18] + 2 * d[5] * d[11] * d[18] + 2 * d[0] * d[12] * d[18] +
        2 * d[1] * d[13] * d[18] + 2 * d[2] * d[14] * d[18] + 2 * d[4] * d[9] * d[19] - 2 * d[3] * d[10] * d[19] -
        2 * d[1] * d[12] * d[19] + 2 * d[0] * d[13] * d[19] + 2 * d[5] * d[9] * d[20] - 2 * d[3] * d[11] * d[20] -
        2 * d[2] * d[12] * d[20] + 2 * d[0] * d[14] * d[20] + 2 * d[0] * d[9] * d[21] - 2 * d[1] * d[10] * d[21] -
        2 * d[2] * d[11] * d[21] + 6 * d[3] * d[12] * d[21] + 2 * d[4] * d[13] * d[21] + 2 * d[5] * d[14] * d[21] +
        2 * d[1] * d[9] * d[22] + 2 * d[0] * d[10] * d[22] + 2 * d[4] * d[12] * d[22] + 2 * d[3] * d[13] * d[22] +
        2 * d[2] * d[9] * d[23] + 2 * d[0] * d[11] * d[23] + 2 * d[5] * d[12] * d[23] + 2 * d[3] * d[14] * d[23];
    coeffs[73] = d[3] * std::pow(d[18], 2) + 2 * d[4] * d[18] * d[19] - d[3] * std::pow(d[19], 2) +
                 2 * d[5] * d[18] * d[20] - d[3] * std::pow(d[20], 2) + 2 * d[0] * d[18] * d[21] -
                 2 * d[1] * d[19] * d[21] - 2 * d[2] * d[20] * d[21] + 3 * d[3] * std::pow(d[21], 2) +
                 2 * d[1] * d[18] * d[22] + 2 * d[0] * d[19] * d[22] + 2 * d[4] * d[21] * d[22] +
                 d[3] * std::pow(d[22], 2) + 2 * d[2] * d[18] * d[23] + 2 * d[0] * d[20] * d[23] +
                 2 * d[5] * d[21] * d[23] + d[3] * std::pow(d[23], 2);
    coeffs[74] = std::pow(d[6], 2) * d[12] - std::pow(d[7], 2) * d[12] - std::pow(d[8], 2) * d[12] +
                 2 * d[6] * d[7] * d[13] + 2 * d[6] * d[8] * d[14] + 2 * d[3] * d[6] * d[15] + 2 * d[4] * d[7] * d[15] +
                 2 * d[5] * d[8] * d[15] + 2 * d[4] * d[6] * d[16] - 2 * d[3] * d[7] * d[16] + 2 * d[5] * d[6] * d[17] -
                 2 * d[3] * d[8] * d[17];
    coeffs[75] = std::pow(d[6], 2) * d[21] - std::pow(d[7], 2) * d[21] - std::pow(d[8], 2) * d[21] +
                 2 * d[6] * d[7] * d[22] + 2 * d[6] * d[8] * d[23] + 2 * d[3] * d[6] * d[24] + 2 * d[4] * d[7] * d[24] +
                 2 * d[5] * d[8] * d[24] + 2 * d[4] * d[6] * d[25] - 2 * d[3] * d[7] * d[25] + 2 * d[5] * d[6] * d[26] -
                 2 * d[3] * d[8] * d[26];
    coeffs[76] = 2 * d[0] * d[3] * d[9] + 2 * d[1] * d[4] * d[9] + 2 * d[2] * d[5] * d[9] - 2 * d[1] * d[3] * d[10] +
                 2 * d[0] * d[4] * d[10] - 2 * d[2] * d[3] * d[11] + 2 * d[0] * d[5] * d[11] +
                 std::pow(d[0], 2) * d[12] - std::pow(d[1], 2) * d[12] - std::pow(d[2], 2) * d[12] +
                 3 * std::pow(d[3], 2) * d[12] + std::pow(d[4], 2) * d[12] + std::pow(d[5], 2) * d[12] +
                 2 * d[0] * d[1] * d[13] + 2 * d[3] * d[4] * d[13] + 2 * d[0] * d[2] * d[14] + 2 * d[3] * d[5] * d[14];
    coeffs[77] = 2 * d[0] * d[3] * d[18] + 2 * d[1] * d[4] * d[18] + 2 * d[2] * d[5] * d[18] - 2 * d[1] * d[3] * d[19] +
                 2 * d[0] * d[4] * d[19] - 2 * d[2] * d[3] * d[20] + 2 * d[0] * d[5] * d[20] +
                 std::pow(d[0], 2) * d[21] - std::pow(d[1], 2) * d[21] - std::pow(d[2], 2) * d[21] +
                 3 * std::pow(d[3], 2) * d[21] + std::pow(d[4], 2) * d[21] + std::pow(d[5], 2) * d[21] +
                 2 * d[0] * d[1] * d[22] + 2 * d[3] * d[4] * d[22] + 2 * d[0] * d[2] * d[23] + 2 * d[3] * d[5] * d[23];
    coeffs[78] = d[3] * std::pow(d[6], 2) + 2 * d[4] * d[6] * d[7] - d[3] * std::pow(d[7], 2) + 2 * d[5] * d[6] * d[8] -
                 d[3] * std::pow(d[8], 2);
    coeffs[79] = std::pow(d[0], 2) * d[3] - std::pow(d[1], 2) * d[3] - std::pow(d[2], 2) * d[3] + std::pow(d[3], 3) +
                 2 * d[0] * d[1] * d[4] + d[3] * std::pow(d[4], 2) + 2 * d[0] * d[2] * d[5] + d[3] * std::pow(d[5], 2);
    coeffs[80] = -d[13] * std::pow(d[15], 2) + 2 * d[12] * d[15] * d[16] + d[13] * std::pow(d[16], 2) +
                 2 * d[14] * d[16] * d[17] - d[13] * std::pow(d[17], 2);
    coeffs[81] = 2 * d[15] * d[16] * d[21] - std::pow(d[15], 2) * d[22] + std::pow(d[16], 2) * d[22] -
                 std::pow(d[17], 2) * d[22] + 2 * d[16] * d[17] * d[23] - 2 * d[13] * d[15] * d[24] +
                 2 * d[12] * d[16] * d[24] + 2 * d[12] * d[15] * d[25] + 2 * d[13] * d[16] * d[25] +
                 2 * d[14] * d[17] * d[25] + 2 * d[14] * d[16] * d[26] - 2 * d[13] * d[17] * d[26];
    coeffs[82] = 2 * d[16] * d[21] * d[24] - 2 * d[15] * d[22] * d[24] - d[13] * std::pow(d[24], 2) +
                 2 * d[15] * d[21] * d[25] + 2 * d[16] * d[22] * d[25] + 2 * d[17] * d[23] * d[25] +
                 2 * d[12] * d[24] * d[25] + d[13] * std::pow(d[25], 2) - 2 * d[17] * d[22] * d[26] +
                 2 * d[16] * d[23] * d[26] + 2 * d[14] * d[25] * d[26] - d[13] * std::pow(d[26], 2);
    coeffs[83] = -d[22] * std::pow(d[24], 2) + 2 * d[21] * d[24] * d[25] + d[22] * std::pow(d[25], 2) +
                 2 * d[23] * d[25] * d[26] - d[22] * std::pow(d[26], 2);
    coeffs[84] = 2 * d[9] * d[10] * d[12] - std::pow(d[9], 2) * d[13] + std::pow(d[10], 2) * d[13] -
                 std::pow(d[11], 2) * d[13] + std::pow(d[12], 2) * d[13] + std::pow(d[13], 3) +
                 2 * d[10] * d[11] * d[14] + d[13] * std::pow(d[14], 2);
    coeffs[85] = 2 * d[10] * d[12] * d[18] - 2 * d[9] * d[13] * d[18] + 2 * d[9] * d[12] * d[19] +
                 2 * d[10] * d[13] * d[19] + 2 * d[11] * d[14] * d[19] - 2 * d[11] * d[13] * d[20] +
                 2 * d[10] * d[14] * d[20] + 2 * d[9] * d[10] * d[21] + 2 * d[12] * d[13] * d[21] -
                 std::pow(d[9], 2) * d[22] + std::pow(d[10], 2) * d[22] - std::pow(d[11], 2) * d[22] +
                 std::pow(d[12], 2) * d[22] + 3 * std::pow(d[13], 2) * d[22] + std::pow(d[14], 2) * d[22] +
                 2 * d[10] * d[11] * d[23] + 2 * d[13] * d[14] * d[23];
    coeffs[86] = -d[13] * std::pow(d[18], 2) + 2 * d[12] * d[18] * d[19] + d[13] * std::pow(d[19], 2) +
                 2 * d[14] * d[19] * d[20] - d[13] * std::pow(d[20], 2) + 2 * d[10] * d[18] * d[21] +
                 2 * d[9] * d[19] * d[21] + d[13] * std::pow(d[21], 2) - 2 * d[9] * d[18] * d[22] +
                 2 * d[10] * d[19] * d[22] - 2 * d[11] * d[20] * d[22] + 2 * d[12] * d[21] * d[22] +
                 3 * d[13] * std::pow(d[22], 2) + 2 * d[11] * d[19] * d[23] + 2 * d[10] * d[20] * d[23] +
                 2 * d[14] * d[22] * d[23] + d[13] * std::pow(d[23], 2);
    coeffs[87] = 2 * d[18] * d[19] * d[21] - std::pow(d[18], 2) * d[22] + std::pow(d[19], 2) * d[22] -
                 std::pow(d[20], 2) * d[22] + std::pow(d[21], 2) * d[22] + std::pow(d[22], 3) +
                 2 * d[19] * d[20] * d[23] + d[22] * std::pow(d[23], 2);
    coeffs[88] = 2 * d[7] * d[12] * d[15] - 2 * d[6] * d[13] * d[15] - d[4] * std::pow(d[15], 2) +
                 2 * d[6] * d[12] * d[16] + 2 * d[7] * d[13] * d[16] + 2 * d[8] * d[14] * d[16] +
                 2 * d[3] * d[15] * d[16] + d[4] * std::pow(d[16], 2) - 2 * d[8] * d[13] * d[17] +
                 2 * d[7] * d[14] * d[17] + 2 * d[5] * d[16] * d[17] - d[4] * std::pow(d[17], 2);
    coeffs[89] = 2 * d[7] * d[15] * d[21] + 2 * d[6] * d[16] * d[21] - 2 * d[6] * d[15] * d[22] +
                 2 * d[7] * d[16] * d[22] - 2 * d[8] * d[17] * d[22] + 2 * d[8] * d[16] * d[23] +
                 2 * d[7] * d[17] * d[23] + 2 * d[7] * d[12] * d[24] - 2 * d[6] * d[13] * d[24] -
                 2 * d[4] * d[15] * d[24] + 2 * d[3] * d[16] * d[24] + 2 * d[6] * d[12] * d[25] +
                 2 * d[7] * d[13] * d[25] + 2 * d[8] * d[14] * d[25] + 2 * d[3] * d[15] * d[25] +
                 2 * d[4] * d[16] * d[25] + 2 * d[5] * d[17] * d[25] - 2 * d[8] * d[13] * d[26] +
                 2 * d[7] * d[14] * d[26] + 2 * d[5] * d[16] * d[26] - 2 * d[4] * d[17] * d[26];
    coeffs[90] = 2 * d[7] * d[21] * d[24] - 2 * d[6] * d[22] * d[24] - d[4] * std::pow(d[24], 2) +
                 2 * d[6] * d[21] * d[25] + 2 * d[7] * d[22] * d[25] + 2 * d[8] * d[23] * d[25] +
                 2 * d[3] * d[24] * d[25] + d[4] * std::pow(d[25], 2) - 2 * d[8] * d[22] * d[26] +
                 2 * d[7] * d[23] * d[26] + 2 * d[5] * d[25] * d[26] - d[4] * std::pow(d[26], 2);
    coeffs[91] = -d[4] * std::pow(d[9], 2) + 2 * d[3] * d[9] * d[10] + d[4] * std::pow(d[10], 2) +
                 2 * d[5] * d[10] * d[11] - d[4] * std::pow(d[11], 2) + 2 * d[1] * d[9] * d[12] +
                 2 * d[0] * d[10] * d[12] + d[4] * std::pow(d[12], 2) - 2 * d[0] * d[9] * d[13] +
                 2 * d[1] * d[10] * d[13] - 2 * d[2] * d[11] * d[13] + 2 * d[3] * d[12] * d[13] +
                 3 * d[4] * std::pow(d[13], 2) + 2 * d[2] * d[10] * d[14] + 2 * d[1] * d[11] * d[14] +
                 2 * d[5] * d[13] * d[14] + d[4] * std::pow(d[14], 2);
    coeffs[92] =
        -2 * d[4] * d[9] * d[18] + 2 * d[3] * d[10] * d[18] + 2 * d[1] * d[12] * d[18] - 2 * d[0] * d[13] * d[18] +
        2 * d[3] * d[9] * d[19] + 2 * d[4] * d[10] * d[19] + 2 * d[5] * d[11] * d[19] + 2 * d[0] * d[12] * d[19] +
        2 * d[1] * d[13] * d[19] + 2 * d[2] * d[14] * d[19] + 2 * d[5] * d[10] * d[20] - 2 * d[4] * d[11] * d[20] -
        2 * d[2] * d[13] * d[20] + 2 * d[1] * d[14] * d[20] + 2 * d[1] * d[9] * d[21] + 2 * d[0] * d[10] * d[21] +
        2 * d[4] * d[12] * d[21] + 2 * d[3] * d[13] * d[21] - 2 * d[0] * d[9] * d[22] + 2 * d[1] * d[10] * d[22] -
        2 * d[2] * d[11] * d[22] + 2 * d[3] * d[12] * d[22] + 6 * d[4] * d[13] * d[22] + 2 * d[5] * d[14] * d[22] +
        2 * d[2] * d[10] * d[23] + 2 * d[1] * d[11] * d[23] + 2 * d[5] * d[13] * d[23] + 2 * d[4] * d[14] * d[23];
    coeffs[93] = -d[4] * std::pow(d[18], 2) + 2 * d[3] * d[18] * d[19] + d[4] * std::pow(d[19], 2) +
                 2 * d[5] * d[19] * d[20] - d[4] * std::pow(d[20], 2) + 2 * d[1] * d[18] * d[21] +
                 2 * d[0] * d[19] * d[21] + d[4] * std::pow(d[21], 2) - 2 * d[0] * d[18] * d[22] +
                 2 * d[1] * d[19] * d[22] - 2 * d[2] * d[20] * d[22] + 2 * d[3] * d[21] * d[22] +
                 3 * d[4] * std::pow(d[22], 2) + 2 * d[2] * d[19] * d[23] + 2 * d[1] * d[20] * d[23] +
                 2 * d[5] * d[22] * d[23] + d[4] * std::pow(d[23], 2);
    coeffs[94] = 2 * d[6] * d[7] * d[12] - std::pow(d[6], 2) * d[13] + std::pow(d[7], 2) * d[13] -
                 std::pow(d[8], 2) * d[13] + 2 * d[7] * d[8] * d[14] - 2 * d[4] * d[6] * d[15] +
                 2 * d[3] * d[7] * d[15] + 2 * d[3] * d[6] * d[16] + 2 * d[4] * d[7] * d[16] + 2 * d[5] * d[8] * d[16] +
                 2 * d[5] * d[7] * d[17] - 2 * d[4] * d[8] * d[17];
    coeffs[95] = 2 * d[6] * d[7] * d[21] - std::pow(d[6], 2) * d[22] + std::pow(d[7], 2) * d[22] -
                 std::pow(d[8], 2) * d[22] + 2 * d[7] * d[8] * d[23] - 2 * d[4] * d[6] * d[24] +
                 2 * d[3] * d[7] * d[24] + 2 * d[3] * d[6] * d[25] + 2 * d[4] * d[7] * d[25] + 2 * d[5] * d[8] * d[25] +
                 2 * d[5] * d[7] * d[26] - 2 * d[4] * d[8] * d[26];
    coeffs[96] = 2 * d[1] * d[3] * d[9] - 2 * d[0] * d[4] * d[9] + 2 * d[0] * d[3] * d[10] + 2 * d[1] * d[4] * d[10] +
                 2 * d[2] * d[5] * d[10] - 2 * d[2] * d[4] * d[11] + 2 * d[1] * d[5] * d[11] + 2 * d[0] * d[1] * d[12] +
                 2 * d[3] * d[4] * d[12] - std::pow(d[0], 2) * d[13] + std::pow(d[1], 2) * d[13] -
                 std::pow(d[2], 2) * d[13] + std::pow(d[3], 2) * d[13] + 3 * std::pow(d[4], 2) * d[13] +
                 std::pow(d[5], 2) * d[13] + 2 * d[1] * d[2] * d[14] + 2 * d[4] * d[5] * d[14];
    coeffs[97] = 2 * d[1] * d[3] * d[18] - 2 * d[0] * d[4] * d[18] + 2 * d[0] * d[3] * d[19] + 2 * d[1] * d[4] * d[19] +
                 2 * d[2] * d[5] * d[19] - 2 * d[2] * d[4] * d[20] + 2 * d[1] * d[5] * d[20] + 2 * d[0] * d[1] * d[21] +
                 2 * d[3] * d[4] * d[21] - std::pow(d[0], 2) * d[22] + std::pow(d[1], 2) * d[22] -
                 std::pow(d[2], 2) * d[22] + std::pow(d[3], 2) * d[22] + 3 * std::pow(d[4], 2) * d[22] +
                 std::pow(d[5], 2) * d[22] + 2 * d[1] * d[2] * d[23] + 2 * d[4] * d[5] * d[23];
    coeffs[98] = -d[4] * std::pow(d[6], 2) + 2 * d[3] * d[6] * d[7] + d[4] * std::pow(d[7], 2) +
                 2 * d[5] * d[7] * d[8] - d[4] * std::pow(d[8], 2);
    coeffs[99] = 2 * d[0] * d[1] * d[3] - std::pow(d[0], 2) * d[4] + std::pow(d[1], 2) * d[4] -
                 std::pow(d[2], 2) * d[4] + std::pow(d[3], 2) * d[4] + std::pow(d[4], 3) + 2 * d[1] * d[2] * d[5] +
                 d[4] * std::pow(d[5], 2);
    coeffs[100] = -d[14] * std::pow(d[15], 2) - d[14] * std::pow(d[16], 2) + 2 * d[12] * d[15] * d[17] +
                  2 * d[13] * d[16] * d[17] + d[14] * std::pow(d[17], 2);
    coeffs[101] = 2 * d[15] * d[17] * d[21] + 2 * d[16] * d[17] * d[22] - std::pow(d[15], 2) * d[23] -
                  std::pow(d[16], 2) * d[23] + std::pow(d[17], 2) * d[23] - 2 * d[14] * d[15] * d[24] +
                  2 * d[12] * d[17] * d[24] - 2 * d[14] * d[16] * d[25] + 2 * d[13] * d[17] * d[25] +
                  2 * d[12] * d[15] * d[26] + 2 * d[13] * d[16] * d[26] + 2 * d[14] * d[17] * d[26];
    coeffs[102] = 2 * d[17] * d[21] * d[24] - 2 * d[15] * d[23] * d[24] - d[14] * std::pow(d[24], 2) +
                  2 * d[17] * d[22] * d[25] - 2 * d[16] * d[23] * d[25] - d[14] * std::pow(d[25], 2) +
                  2 * d[15] * d[21] * d[26] + 2 * d[16] * d[22] * d[26] + 2 * d[17] * d[23] * d[26] +
                  2 * d[12] * d[24] * d[26] + 2 * d[13] * d[25] * d[26] + d[14] * std::pow(d[26], 2);
    coeffs[103] = -d[23] * std::pow(d[24], 2) - d[23] * std::pow(d[25], 2) + 2 * d[21] * d[24] * d[26] +
                  2 * d[22] * d[25] * d[26] + d[23] * std::pow(d[26], 2);
    coeffs[104] = 2 * d[9] * d[11] * d[12] + 2 * d[10] * d[11] * d[13] - std::pow(d[9], 2) * d[14] -
                  std::pow(d[10], 2) * d[14] + std::pow(d[11], 2) * d[14] + std::pow(d[12], 2) * d[14] +
                  std::pow(d[13], 2) * d[14] + std::pow(d[14], 3);
    coeffs[105] = 2 * d[11] * d[12] * d[18] - 2 * d[9] * d[14] * d[18] + 2 * d[11] * d[13] * d[19] -
                  2 * d[10] * d[14] * d[19] + 2 * d[9] * d[12] * d[20] + 2 * d[10] * d[13] * d[20] +
                  2 * d[11] * d[14] * d[20] + 2 * d[9] * d[11] * d[21] + 2 * d[12] * d[14] * d[21] +
                  2 * d[10] * d[11] * d[22] + 2 * d[13] * d[14] * d[22] - std::pow(d[9], 2) * d[23] -
                  std::pow(d[10], 2) * d[23] + std::pow(d[11], 2) * d[23] + std::pow(d[12], 2) * d[23] +
                  std::pow(d[13], 2) * d[23] + 3 * std::pow(d[14], 2) * d[23];
    coeffs[106] = -d[14] * std::pow(d[18], 2) - d[14] * std::pow(d[19], 2) + 2 * d[12] * d[18] * d[20] +
                  2 * d[13] * d[19] * d[20] + d[14] * std::pow(d[20], 2) + 2 * d[11] * d[18] * d[21] +
                  2 * d[9] * d[20] * d[21] + d[14] * std::pow(d[21], 2) + 2 * d[11] * d[19] * d[22] +
                  2 * d[10] * d[20] * d[22] + d[14] * std::pow(d[22], 2) - 2 * d[9] * d[18] * d[23] -
                  2 * d[10] * d[19] * d[23] + 2 * d[11] * d[20] * d[23] + 2 * d[12] * d[21] * d[23] +
                  2 * d[13] * d[22] * d[23] + 3 * d[14] * std::pow(d[23], 2);
    coeffs[107] = 2 * d[18] * d[20] * d[21] + 2 * d[19] * d[20] * d[22] - std::pow(d[18], 2) * d[23] -
                  std::pow(d[19], 2) * d[23] + std::pow(d[20], 2) * d[23] + std::pow(d[21], 2) * d[23] +
                  std::pow(d[22], 2) * d[23] + std::pow(d[23], 3);
    coeffs[108] = 2 * d[8] * d[12] * d[15] - 2 * d[6] * d[14] * d[15] - d[5] * std::pow(d[15], 2) +
                  2 * d[8] * d[13] * d[16] - 2 * d[7] * d[14] * d[16] - d[5] * std::pow(d[16], 2) +
                  2 * d[6] * d[12] * d[17] + 2 * d[7] * d[13] * d[17] + 2 * d[8] * d[14] * d[17] +
                  2 * d[3] * d[15] * d[17] + 2 * d[4] * d[16] * d[17] + d[5] * std::pow(d[17], 2);
    coeffs[109] = 2 * d[8] * d[15] * d[21] + 2 * d[6] * d[17] * d[21] + 2 * d[8] * d[16] * d[22] +
                  2 * d[7] * d[17] * d[22] - 2 * d[6] * d[15] * d[23] - 2 * d[7] * d[16] * d[23] +
                  2 * d[8] * d[17] * d[23] + 2 * d[8] * d[12] * d[24] - 2 * d[6] * d[14] * d[24] -
                  2 * d[5] * d[15] * d[24] + 2 * d[3] * d[17] * d[24] + 2 * d[8] * d[13] * d[25] -
                  2 * d[7] * d[14] * d[25] - 2 * d[5] * d[16] * d[25] + 2 * d[4] * d[17] * d[25] +
                  2 * d[6] * d[12] * d[26] + 2 * d[7] * d[13] * d[26] + 2 * d[8] * d[14] * d[26] +
                  2 * d[3] * d[15] * d[26] + 2 * d[4] * d[16] * d[26] + 2 * d[5] * d[17] * d[26];
    coeffs[110] = 2 * d[8] * d[21] * d[24] - 2 * d[6] * d[23] * d[24] - d[5] * std::pow(d[24], 2) +
                  2 * d[8] * d[22] * d[25] - 2 * d[7] * d[23] * d[25] - d[5] * std::pow(d[25], 2) +
                  2 * d[6] * d[21] * d[26] + 2 * d[7] * d[22] * d[26] + 2 * d[8] * d[23] * d[26] +
                  2 * d[3] * d[24] * d[26] + 2 * d[4] * d[25] * d[26] + d[5] * std::pow(d[26], 2);
    coeffs[111] = -d[5] * std::pow(d[9], 2) - d[5] * std::pow(d[10], 2) + 2 * d[3] * d[9] * d[11] +
                  2 * d[4] * d[10] * d[11] + d[5] * std::pow(d[11], 2) + 2 * d[2] * d[9] * d[12] +
                  2 * d[0] * d[11] * d[12] + d[5] * std::pow(d[12], 2) + 2 * d[2] * d[10] * d[13] +
                  2 * d[1] * d[11] * d[13] + d[5] * std::pow(d[13], 2) - 2 * d[0] * d[9] * d[14] -
                  2 * d[1] * d[10] * d[14] + 2 * d[2] * d[11] * d[14] + 2 * d[3] * d[12] * d[14] +
                  2 * d[4] * d[13] * d[14] + 3 * d[5] * std::pow(d[14], 2);
    coeffs[112] =
        -2 * d[5] * d[9] * d[18] + 2 * d[3] * d[11] * d[18] + 2 * d[2] * d[12] * d[18] - 2 * d[0] * d[14] * d[18] -
        2 * d[5] * d[10] * d[19] + 2 * d[4] * d[11] * d[19] + 2 * d[2] * d[13] * d[19] - 2 * d[1] * d[14] * d[19] +
        2 * d[3] * d[9] * d[20] + 2 * d[4] * d[10] * d[20] + 2 * d[5] * d[11] * d[20] + 2 * d[0] * d[12] * d[20] +
        2 * d[1] * d[13] * d[20] + 2 * d[2] * d[14] * d[20] + 2 * d[2] * d[9] * d[21] + 2 * d[0] * d[11] * d[21] +
        2 * d[5] * d[12] * d[21] + 2 * d[3] * d[14] * d[21] + 2 * d[2] * d[10] * d[22] + 2 * d[1] * d[11] * d[22] +
        2 * d[5] * d[13] * d[22] + 2 * d[4] * d[14] * d[22] - 2 * d[0] * d[9] * d[23] - 2 * d[1] * d[10] * d[23] +
        2 * d[2] * d[11] * d[23] + 2 * d[3] * d[12] * d[23] + 2 * d[4] * d[13] * d[23] + 6 * d[5] * d[14] * d[23];
    coeffs[113] = -d[5] * std::pow(d[18], 2) - d[5] * std::pow(d[19], 2) + 2 * d[3] * d[18] * d[20] +
                  2 * d[4] * d[19] * d[20] + d[5] * std::pow(d[20], 2) + 2 * d[2] * d[18] * d[21] +
                  2 * d[0] * d[20] * d[21] + d[5] * std::pow(d[21], 2) + 2 * d[2] * d[19] * d[22] +
                  2 * d[1] * d[20] * d[22] + d[5] * std::pow(d[22], 2) - 2 * d[0] * d[18] * d[23] -
                  2 * d[1] * d[19] * d[23] + 2 * d[2] * d[20] * d[23] + 2 * d[3] * d[21] * d[23] +
                  2 * d[4] * d[22] * d[23] + 3 * d[5] * std::pow(d[23], 2);
    coeffs[114] = 2 * d[6] * d[8] * d[12] + 2 * d[7] * d[8] * d[13] - std::pow(d[6], 2) * d[14] -
                  std::pow(d[7], 2) * d[14] + std::pow(d[8], 2) * d[14] - 2 * d[5] * d[6] * d[15] +
                  2 * d[3] * d[8] * d[15] - 2 * d[5] * d[7] * d[16] + 2 * d[4] * d[8] * d[16] +
                  2 * d[3] * d[6] * d[17] + 2 * d[4] * d[7] * d[17] + 2 * d[5] * d[8] * d[17];
    coeffs[115] = 2 * d[6] * d[8] * d[21] + 2 * d[7] * d[8] * d[22] - std::pow(d[6], 2) * d[23] -
                  std::pow(d[7], 2) * d[23] + std::pow(d[8], 2) * d[23] - 2 * d[5] * d[6] * d[24] +
                  2 * d[3] * d[8] * d[24] - 2 * d[5] * d[7] * d[25] + 2 * d[4] * d[8] * d[25] +
                  2 * d[3] * d[6] * d[26] + 2 * d[4] * d[7] * d[26] + 2 * d[5] * d[8] * d[26];
    coeffs[116] = 2 * d[2] * d[3] * d[9] - 2 * d[0] * d[5] * d[9] + 2 * d[2] * d[4] * d[10] - 2 * d[1] * d[5] * d[10] +
                  2 * d[0] * d[3] * d[11] + 2 * d[1] * d[4] * d[11] + 2 * d[2] * d[5] * d[11] +
                  2 * d[0] * d[2] * d[12] + 2 * d[3] * d[5] * d[12] + 2 * d[1] * d[2] * d[13] +
                  2 * d[4] * d[5] * d[13] - std::pow(d[0], 2) * d[14] - std::pow(d[1], 2) * d[14] +
                  std::pow(d[2], 2) * d[14] + std::pow(d[3], 2) * d[14] + std::pow(d[4], 2) * d[14] +
                  3 * std::pow(d[5], 2) * d[14];
    coeffs[117] = 2 * d[2] * d[3] * d[18] - 2 * d[0] * d[5] * d[18] + 2 * d[2] * d[4] * d[19] -
                  2 * d[1] * d[5] * d[19] + 2 * d[0] * d[3] * d[20] + 2 * d[1] * d[4] * d[20] +
                  2 * d[2] * d[5] * d[20] + 2 * d[0] * d[2] * d[21] + 2 * d[3] * d[5] * d[21] +
                  2 * d[1] * d[2] * d[22] + 2 * d[4] * d[5] * d[22] - std::pow(d[0], 2) * d[23] -
                  std::pow(d[1], 2) * d[23] + std::pow(d[2], 2) * d[23] + std::pow(d[3], 2) * d[23] +
                  std::pow(d[4], 2) * d[23] + 3 * std::pow(d[5], 2) * d[23];
    coeffs[118] = -d[5] * std::pow(d[6], 2) - d[5] * std::pow(d[7], 2) + 2 * d[3] * d[6] * d[8] +
                  2 * d[4] * d[7] * d[8] + d[5] * std::pow(d[8], 2);
    coeffs[119] = 2 * d[0] * d[2] * d[3] + 2 * d[1] * d[2] * d[4] - std::pow(d[0], 2) * d[5] -
                  std::pow(d[1], 2) * d[5] + std::pow(d[2], 2) * d[5] + std::pow(d[3], 2) * d[5] +
                  std::pow(d[4], 2) * d[5] + std::pow(d[5], 3);
    coeffs[120] = std::pow(d[15], 3) + d[15] * std::pow(d[16], 2) + d[15] * std::pow(d[17], 2);
    coeffs[121] = 3 * std::pow(d[15], 2) * d[24] + std::pow(d[16], 2) * d[24] + std::pow(d[17], 2) * d[24] +
                  2 * d[15] * d[16] * d[25] + 2 * d[15] * d[17] * d[26];
    coeffs[122] = 3 * d[15] * std::pow(d[24], 2) + 2 * d[16] * d[24] * d[25] + d[15] * std::pow(d[25], 2) +
                  2 * d[17] * d[24] * d[26] + d[15] * std::pow(d[26], 2);
    coeffs[123] = std::pow(d[24], 3) + d[24] * std::pow(d[25], 2) + d[24] * std::pow(d[26], 2);
    coeffs[124] = std::pow(d[9], 2) * d[15] - std::pow(d[10], 2) * d[15] - std::pow(d[11], 2) * d[15] +
                  std::pow(d[12], 2) * d[15] - std::pow(d[13], 2) * d[15] - std::pow(d[14], 2) * d[15] +
                  2 * d[9] * d[10] * d[16] + 2 * d[12] * d[13] * d[16] + 2 * d[9] * d[11] * d[17] +
                  2 * d[12] * d[14] * d[17];
    coeffs[125] = 2 * d[9] * d[15] * d[18] + 2 * d[10] * d[16] * d[18] + 2 * d[11] * d[17] * d[18] -
                  2 * d[10] * d[15] * d[19] + 2 * d[9] * d[16] * d[19] - 2 * d[11] * d[15] * d[20] +
                  2 * d[9] * d[17] * d[20] + 2 * d[12] * d[15] * d[21] + 2 * d[13] * d[16] * d[21] +
                  2 * d[14] * d[17] * d[21] - 2 * d[13] * d[15] * d[22] + 2 * d[12] * d[16] * d[22] -
                  2 * d[14] * d[15] * d[23] + 2 * d[12] * d[17] * d[23] + std::pow(d[9], 2) * d[24] -
                  std::pow(d[10], 2) * d[24] - std::pow(d[11], 2) * d[24] + std::pow(d[12], 2) * d[24] -
                  std::pow(d[13], 2) * d[24] - std::pow(d[14], 2) * d[24] + 2 * d[9] * d[10] * d[25] +
                  2 * d[12] * d[13] * d[25] + 2 * d[9] * d[11] * d[26] + 2 * d[12] * d[14] * d[26];
    coeffs[126] = d[15] * std::pow(d[18], 2) + 2 * d[16] * d[18] * d[19] - d[15] * std::pow(d[19], 2) +
                  2 * d[17] * d[18] * d[20] - d[15] * std::pow(d[20], 2) + d[15] * std::pow(d[21], 2) +
                  2 * d[16] * d[21] * d[22] - d[15] * std::pow(d[22], 2) + 2 * d[17] * d[21] * d[23] -
                  d[15] * std::pow(d[23], 2) + 2 * d[9] * d[18] * d[24] - 2 * d[10] * d[19] * d[24] -
                  2 * d[11] * d[20] * d[24] + 2 * d[12] * d[21] * d[24] - 2 * d[13] * d[22] * d[24] -
                  2 * d[14] * d[23] * d[24] + 2 * d[10] * d[18] * d[25] + 2 * d[9] * d[19] * d[25] +
                  2 * d[13] * d[21] * d[25] + 2 * d[12] * d[22] * d[25] + 2 * d[11] * d[18] * d[26] +
                  2 * d[9] * d[20] * d[26] + 2 * d[14] * d[21] * d[26] + 2 * d[12] * d[23] * d[26];
    coeffs[127] = std::pow(d[18], 2) * d[24] - std::pow(d[19], 2) * d[24] - std::pow(d[20], 2) * d[24] +
                  std::pow(d[21], 2) * d[24] - std::pow(d[22], 2) * d[24] - std::pow(d[23], 2) * d[24] +
                  2 * d[18] * d[19] * d[25] + 2 * d[21] * d[22] * d[25] + 2 * d[18] * d[20] * d[26] +
                  2 * d[21] * d[23] * d[26];
    coeffs[128] = 3 * d[6] * std::pow(d[15], 2) + 2 * d[7] * d[15] * d[16] + d[6] * std::pow(d[16], 2) +
                  2 * d[8] * d[15] * d[17] + d[6] * std::pow(d[17], 2);
    coeffs[129] = 6 * d[6] * d[15] * d[24] + 2 * d[7] * d[16] * d[24] + 2 * d[8] * d[17] * d[24] +
                  2 * d[7] * d[15] * d[25] + 2 * d[6] * d[16] * d[25] + 2 * d[8] * d[15] * d[26] +
                  2 * d[6] * d[17] * d[26];
    coeffs[130] = 3 * d[6] * std::pow(d[24], 2) + 2 * d[7] * d[24] * d[25] + d[6] * std::pow(d[25], 2) +
                  2 * d[8] * d[24] * d[26] + d[6] * std::pow(d[26], 2);
    coeffs[131] =
        d[6] * std::pow(d[9], 2) + 2 * d[7] * d[9] * d[10] - d[6] * std::pow(d[10], 2) + 2 * d[8] * d[9] * d[11] -
        d[6] * std::pow(d[11], 2) + d[6] * std::pow(d[12], 2) + 2 * d[7] * d[12] * d[13] - d[6] * std::pow(d[13], 2) +
        2 * d[8] * d[12] * d[14] - d[6] * std::pow(d[14], 2) + 2 * d[0] * d[9] * d[15] - 2 * d[1] * d[10] * d[15] -
        2 * d[2] * d[11] * d[15] + 2 * d[3] * d[12] * d[15] - 2 * d[4] * d[13] * d[15] - 2 * d[5] * d[14] * d[15] +
        2 * d[1] * d[9] * d[16] + 2 * d[0] * d[10] * d[16] + 2 * d[4] * d[12] * d[16] + 2 * d[3] * d[13] * d[16] +
        2 * d[2] * d[9] * d[17] + 2 * d[0] * d[11] * d[17] + 2 * d[5] * d[12] * d[17] + 2 * d[3] * d[14] * d[17];
    coeffs[132] =
        2 * d[6] * d[9] * d[18] + 2 * d[7] * d[10] * d[18] + 2 * d[8] * d[11] * d[18] + 2 * d[0] * d[15] * d[18] +
        2 * d[1] * d[16] * d[18] + 2 * d[2] * d[17] * d[18] + 2 * d[7] * d[9] * d[19] - 2 * d[6] * d[10] * d[19] -
        2 * d[1] * d[15] * d[19] + 2 * d[0] * d[16] * d[19] + 2 * d[8] * d[9] * d[20] - 2 * d[6] * d[11] * d[20] -
        2 * d[2] * d[15] * d[20] + 2 * d[0] * d[17] * d[20] + 2 * d[6] * d[12] * d[21] + 2 * d[7] * d[13] * d[21] +
        2 * d[8] * d[14] * d[21] + 2 * d[3] * d[15] * d[21] + 2 * d[4] * d[16] * d[21] + 2 * d[5] * d[17] * d[21] +
        2 * d[7] * d[12] * d[22] - 2 * d[6] * d[13] * d[22] - 2 * d[4] * d[15] * d[22] + 2 * d[3] * d[16] * d[22] +
        2 * d[8] * d[12] * d[23] - 2 * d[6] * d[14] * d[23] - 2 * d[5] * d[15] * d[23] + 2 * d[3] * d[17] * d[23] +
        2 * d[0] * d[9] * d[24] - 2 * d[1] * d[10] * d[24] - 2 * d[2] * d[11] * d[24] + 2 * d[3] * d[12] * d[24] -
        2 * d[4] * d[13] * d[24] - 2 * d[5] * d[14] * d[24] + 2 * d[1] * d[9] * d[25] + 2 * d[0] * d[10] * d[25] +
        2 * d[4] * d[12] * d[25] + 2 * d[3] * d[13] * d[25] + 2 * d[2] * d[9] * d[26] + 2 * d[0] * d[11] * d[26] +
        2 * d[5] * d[12] * d[26] + 2 * d[3] * d[14] * d[26];
    coeffs[133] =
        d[6] * std::pow(d[18], 2) + 2 * d[7] * d[18] * d[19] - d[6] * std::pow(d[19], 2) + 2 * d[8] * d[18] * d[20] -
        d[6] * std::pow(d[20], 2) + d[6] * std::pow(d[21], 2) + 2 * d[7] * d[21] * d[22] - d[6] * std::pow(d[22], 2) +
        2 * d[8] * d[21] * d[23] - d[6] * std::pow(d[23], 2) + 2 * d[0] * d[18] * d[24] - 2 * d[1] * d[19] * d[24] -
        2 * d[2] * d[20] * d[24] + 2 * d[3] * d[21] * d[24] - 2 * d[4] * d[22] * d[24] - 2 * d[5] * d[23] * d[24] +
        2 * d[1] * d[18] * d[25] + 2 * d[0] * d[19] * d[25] + 2 * d[4] * d[21] * d[25] + 2 * d[3] * d[22] * d[25] +
        2 * d[2] * d[18] * d[26] + 2 * d[0] * d[20] * d[26] + 2 * d[5] * d[21] * d[26] + 2 * d[3] * d[23] * d[26];
    coeffs[134] = 3 * std::pow(d[6], 2) * d[15] + std::pow(d[7], 2) * d[15] + std::pow(d[8], 2) * d[15] +
                  2 * d[6] * d[7] * d[16] + 2 * d[6] * d[8] * d[17];
    coeffs[135] = 3 * std::pow(d[6], 2) * d[24] + std::pow(d[7], 2) * d[24] + std::pow(d[8], 2) * d[24] +
                  2 * d[6] * d[7] * d[25] + 2 * d[6] * d[8] * d[26];
    coeffs[136] =
        2 * d[0] * d[6] * d[9] + 2 * d[1] * d[7] * d[9] + 2 * d[2] * d[8] * d[9] - 2 * d[1] * d[6] * d[10] +
        2 * d[0] * d[7] * d[10] - 2 * d[2] * d[6] * d[11] + 2 * d[0] * d[8] * d[11] + 2 * d[3] * d[6] * d[12] +
        2 * d[4] * d[7] * d[12] + 2 * d[5] * d[8] * d[12] - 2 * d[4] * d[6] * d[13] + 2 * d[3] * d[7] * d[13] -
        2 * d[5] * d[6] * d[14] + 2 * d[3] * d[8] * d[14] + std::pow(d[0], 2) * d[15] - std::pow(d[1], 2) * d[15] -
        std::pow(d[2], 2) * d[15] + std::pow(d[3], 2) * d[15] - std::pow(d[4], 2) * d[15] - std::pow(d[5], 2) * d[15] +
        2 * d[0] * d[1] * d[16] + 2 * d[3] * d[4] * d[16] + 2 * d[0] * d[2] * d[17] + 2 * d[3] * d[5] * d[17];
    coeffs[137] =
        2 * d[0] * d[6] * d[18] + 2 * d[1] * d[7] * d[18] + 2 * d[2] * d[8] * d[18] - 2 * d[1] * d[6] * d[19] +
        2 * d[0] * d[7] * d[19] - 2 * d[2] * d[6] * d[20] + 2 * d[0] * d[8] * d[20] + 2 * d[3] * d[6] * d[21] +
        2 * d[4] * d[7] * d[21] + 2 * d[5] * d[8] * d[21] - 2 * d[4] * d[6] * d[22] + 2 * d[3] * d[7] * d[22] -
        2 * d[5] * d[6] * d[23] + 2 * d[3] * d[8] * d[23] + std::pow(d[0], 2) * d[24] - std::pow(d[1], 2) * d[24] -
        std::pow(d[2], 2) * d[24] + std::pow(d[3], 2) * d[24] - std::pow(d[4], 2) * d[24] - std::pow(d[5], 2) * d[24] +
        2 * d[0] * d[1] * d[25] + 2 * d[3] * d[4] * d[25] + 2 * d[0] * d[2] * d[26] + 2 * d[3] * d[5] * d[26];
    coeffs[138] = std::pow(d[6], 3) + d[6] * std::pow(d[7], 2) + d[6] * std::pow(d[8], 2);
    coeffs[139] = std::pow(d[0], 2) * d[6] - std::pow(d[1], 2) * d[6] - std::pow(d[2], 2) * d[6] +
                  std::pow(d[3], 2) * d[6] - std::pow(d[4], 2) * d[6] - std::pow(d[5], 2) * d[6] +
                  2 * d[0] * d[1] * d[7] + 2 * d[3] * d[4] * d[7] + 2 * d[0] * d[2] * d[8] + 2 * d[3] * d[5] * d[8];
    coeffs[140] = std::pow(d[15], 2) * d[16] + std::pow(d[16], 3) + d[16] * std::pow(d[17], 2);
    coeffs[141] = 2 * d[15] * d[16] * d[24] + std::pow(d[15], 2) * d[25] + 3 * std::pow(d[16], 2) * d[25] +
                  std::pow(d[17], 2) * d[25] + 2 * d[16] * d[17] * d[26];
    coeffs[142] = d[16] * std::pow(d[24], 2) + 2 * d[15] * d[24] * d[25] + 3 * d[16] * std::pow(d[25], 2) +
                  2 * d[17] * d[25] * d[26] + d[16] * std::pow(d[26], 2);
    coeffs[143] = std::pow(d[24], 2) * d[25] + std::pow(d[25], 3) + d[25] * std::pow(d[26], 2);
    coeffs[144] = 2 * d[9] * d[10] * d[15] + 2 * d[12] * d[13] * d[15] - std::pow(d[9], 2) * d[16] +
                  std::pow(d[10], 2) * d[16] - std::pow(d[11], 2) * d[16] - std::pow(d[12], 2) * d[16] +
                  std::pow(d[13], 2) * d[16] - std::pow(d[14], 2) * d[16] + 2 * d[10] * d[11] * d[17] +
                  2 * d[13] * d[14] * d[17];
    coeffs[145] = 2 * d[10] * d[15] * d[18] - 2 * d[9] * d[16] * d[18] + 2 * d[9] * d[15] * d[19] +
                  2 * d[10] * d[16] * d[19] + 2 * d[11] * d[17] * d[19] - 2 * d[11] * d[16] * d[20] +
                  2 * d[10] * d[17] * d[20] + 2 * d[13] * d[15] * d[21] - 2 * d[12] * d[16] * d[21] +
                  2 * d[12] * d[15] * d[22] + 2 * d[13] * d[16] * d[22] + 2 * d[14] * d[17] * d[22] -
                  2 * d[14] * d[16] * d[23] + 2 * d[13] * d[17] * d[23] + 2 * d[9] * d[10] * d[24] +
                  2 * d[12] * d[13] * d[24] - std::pow(d[9], 2) * d[25] + std::pow(d[10], 2) * d[25] -
                  std::pow(d[11], 2) * d[25] - std::pow(d[12], 2) * d[25] + std::pow(d[13], 2) * d[25] -
                  std::pow(d[14], 2) * d[25] + 2 * d[10] * d[11] * d[26] + 2 * d[13] * d[14] * d[26];
    coeffs[146] = -d[16] * std::pow(d[18], 2) + 2 * d[15] * d[18] * d[19] + d[16] * std::pow(d[19], 2) +
                  2 * d[17] * d[19] * d[20] - d[16] * std::pow(d[20], 2) - d[16] * std::pow(d[21], 2) +
                  2 * d[15] * d[21] * d[22] + d[16] * std::pow(d[22], 2) + 2 * d[17] * d[22] * d[23] -
                  d[16] * std::pow(d[23], 2) + 2 * d[10] * d[18] * d[24] + 2 * d[9] * d[19] * d[24] +
                  2 * d[13] * d[21] * d[24] + 2 * d[12] * d[22] * d[24] - 2 * d[9] * d[18] * d[25] +
                  2 * d[10] * d[19] * d[25] - 2 * d[11] * d[20] * d[25] - 2 * d[12] * d[21] * d[25] +
                  2 * d[13] * d[22] * d[25] - 2 * d[14] * d[23] * d[25] + 2 * d[11] * d[19] * d[26] +
                  2 * d[10] * d[20] * d[26] + 2 * d[14] * d[22] * d[26] + 2 * d[13] * d[23] * d[26];
    coeffs[147] = 2 * d[18] * d[19] * d[24] + 2 * d[21] * d[22] * d[24] - std::pow(d[18], 2) * d[25] +
                  std::pow(d[19], 2) * d[25] - std::pow(d[20], 2) * d[25] - std::pow(d[21], 2) * d[25] +
                  std::pow(d[22], 2) * d[25] - std::pow(d[23], 2) * d[25] + 2 * d[19] * d[20] * d[26] +
                  2 * d[22] * d[23] * d[26];
    coeffs[148] = d[7] * std::pow(d[15], 2) + 2 * d[6] * d[15] * d[16] + 3 * d[7] * std::pow(d[16], 2) +
                  2 * d[8] * d[16] * d[17] + d[7] * std::pow(d[17], 2);
    coeffs[149] = 2 * d[7] * d[15] * d[24] + 2 * d[6] * d[16] * d[24] + 2 * d[6] * d[15] * d[25] +
                  6 * d[7] * d[16] * d[25] + 2 * d[8] * d[17] * d[25] + 2 * d[8] * d[16] * d[26] +
                  2 * d[7] * d[17] * d[26];
    coeffs[150] = d[7] * std::pow(d[24], 2) + 2 * d[6] * d[24] * d[25] + 3 * d[7] * std::pow(d[25], 2) +
                  2 * d[8] * d[25] * d[26] + d[7] * std::pow(d[26], 2);
    coeffs[151] =
        -d[7] * std::pow(d[9], 2) + 2 * d[6] * d[9] * d[10] + d[7] * std::pow(d[10], 2) + 2 * d[8] * d[10] * d[11] -
        d[7] * std::pow(d[11], 2) - d[7] * std::pow(d[12], 2) + 2 * d[6] * d[12] * d[13] + d[7] * std::pow(d[13], 2) +
        2 * d[8] * d[13] * d[14] - d[7] * std::pow(d[14], 2) + 2 * d[1] * d[9] * d[15] + 2 * d[0] * d[10] * d[15] +
        2 * d[4] * d[12] * d[15] + 2 * d[3] * d[13] * d[15] - 2 * d[0] * d[9] * d[16] + 2 * d[1] * d[10] * d[16] -
        2 * d[2] * d[11] * d[16] - 2 * d[3] * d[12] * d[16] + 2 * d[4] * d[13] * d[16] - 2 * d[5] * d[14] * d[16] +
        2 * d[2] * d[10] * d[17] + 2 * d[1] * d[11] * d[17] + 2 * d[5] * d[13] * d[17] + 2 * d[4] * d[14] * d[17];
    coeffs[152] =
        -2 * d[7] * d[9] * d[18] + 2 * d[6] * d[10] * d[18] + 2 * d[1] * d[15] * d[18] - 2 * d[0] * d[16] * d[18] +
        2 * d[6] * d[9] * d[19] + 2 * d[7] * d[10] * d[19] + 2 * d[8] * d[11] * d[19] + 2 * d[0] * d[15] * d[19] +
        2 * d[1] * d[16] * d[19] + 2 * d[2] * d[17] * d[19] + 2 * d[8] * d[10] * d[20] - 2 * d[7] * d[11] * d[20] -
        2 * d[2] * d[16] * d[20] + 2 * d[1] * d[17] * d[20] - 2 * d[7] * d[12] * d[21] + 2 * d[6] * d[13] * d[21] +
        2 * d[4] * d[15] * d[21] - 2 * d[3] * d[16] * d[21] + 2 * d[6] * d[12] * d[22] + 2 * d[7] * d[13] * d[22] +
        2 * d[8] * d[14] * d[22] + 2 * d[3] * d[15] * d[22] + 2 * d[4] * d[16] * d[22] + 2 * d[5] * d[17] * d[22] +
        2 * d[8] * d[13] * d[23] - 2 * d[7] * d[14] * d[23] - 2 * d[5] * d[16] * d[23] + 2 * d[4] * d[17] * d[23] +
        2 * d[1] * d[9] * d[24] + 2 * d[0] * d[10] * d[24] + 2 * d[4] * d[12] * d[24] + 2 * d[3] * d[13] * d[24] -
        2 * d[0] * d[9] * d[25] + 2 * d[1] * d[10] * d[25] - 2 * d[2] * d[11] * d[25] - 2 * d[3] * d[12] * d[25] +
        2 * d[4] * d[13] * d[25] - 2 * d[5] * d[14] * d[25] + 2 * d[2] * d[10] * d[26] + 2 * d[1] * d[11] * d[26] +
        2 * d[5] * d[13] * d[26] + 2 * d[4] * d[14] * d[26];
    coeffs[153] =
        -d[7] * std::pow(d[18], 2) + 2 * d[6] * d[18] * d[19] + d[7] * std::pow(d[19], 2) + 2 * d[8] * d[19] * d[20] -
        d[7] * std::pow(d[20], 2) - d[7] * std::pow(d[21], 2) + 2 * d[6] * d[21] * d[22] + d[7] * std::pow(d[22], 2) +
        2 * d[8] * d[22] * d[23] - d[7] * std::pow(d[23], 2) + 2 * d[1] * d[18] * d[24] + 2 * d[0] * d[19] * d[24] +
        2 * d[4] * d[21] * d[24] + 2 * d[3] * d[22] * d[24] - 2 * d[0] * d[18] * d[25] + 2 * d[1] * d[19] * d[25] -
        2 * d[2] * d[20] * d[25] - 2 * d[3] * d[21] * d[25] + 2 * d[4] * d[22] * d[25] - 2 * d[5] * d[23] * d[25] +
        2 * d[2] * d[19] * d[26] + 2 * d[1] * d[20] * d[26] + 2 * d[5] * d[22] * d[26] + 2 * d[4] * d[23] * d[26];
    coeffs[154] = 2 * d[6] * d[7] * d[15] + std::pow(d[6], 2) * d[16] + 3 * std::pow(d[7], 2) * d[16] +
                  std::pow(d[8], 2) * d[16] + 2 * d[7] * d[8] * d[17];
    coeffs[155] = 2 * d[6] * d[7] * d[24] + std::pow(d[6], 2) * d[25] + 3 * std::pow(d[7], 2) * d[25] +
                  std::pow(d[8], 2) * d[25] + 2 * d[7] * d[8] * d[26];
    coeffs[156] =
        2 * d[1] * d[6] * d[9] - 2 * d[0] * d[7] * d[9] + 2 * d[0] * d[6] * d[10] + 2 * d[1] * d[7] * d[10] +
        2 * d[2] * d[8] * d[10] - 2 * d[2] * d[7] * d[11] + 2 * d[1] * d[8] * d[11] + 2 * d[4] * d[6] * d[12] -
        2 * d[3] * d[7] * d[12] + 2 * d[3] * d[6] * d[13] + 2 * d[4] * d[7] * d[13] + 2 * d[5] * d[8] * d[13] -
        2 * d[5] * d[7] * d[14] + 2 * d[4] * d[8] * d[14] + 2 * d[0] * d[1] * d[15] + 2 * d[3] * d[4] * d[15] -
        std::pow(d[0], 2) * d[16] + std::pow(d[1], 2) * d[16] - std::pow(d[2], 2) * d[16] - std::pow(d[3], 2) * d[16] +
        std::pow(d[4], 2) * d[16] - std::pow(d[5], 2) * d[16] + 2 * d[1] * d[2] * d[17] + 2 * d[4] * d[5] * d[17];
    coeffs[157] =
        2 * d[1] * d[6] * d[18] - 2 * d[0] * d[7] * d[18] + 2 * d[0] * d[6] * d[19] + 2 * d[1] * d[7] * d[19] +
        2 * d[2] * d[8] * d[19] - 2 * d[2] * d[7] * d[20] + 2 * d[1] * d[8] * d[20] + 2 * d[4] * d[6] * d[21] -
        2 * d[3] * d[7] * d[21] + 2 * d[3] * d[6] * d[22] + 2 * d[4] * d[7] * d[22] + 2 * d[5] * d[8] * d[22] -
        2 * d[5] * d[7] * d[23] + 2 * d[4] * d[8] * d[23] + 2 * d[0] * d[1] * d[24] + 2 * d[3] * d[4] * d[24] -
        std::pow(d[0], 2) * d[25] + std::pow(d[1], 2) * d[25] - std::pow(d[2], 2) * d[25] - std::pow(d[3], 2) * d[25] +
        std::pow(d[4], 2) * d[25] - std::pow(d[5], 2) * d[25] + 2 * d[1] * d[2] * d[26] + 2 * d[4] * d[5] * d[26];
    coeffs[158] = std::pow(d[6], 2) * d[7] + std::pow(d[7], 3) + d[7] * std::pow(d[8], 2);
    coeffs[159] = 2 * d[0] * d[1] * d[6] + 2 * d[3] * d[4] * d[6] - std::pow(d[0], 2) * d[7] +
                  std::pow(d[1], 2) * d[7] - std::pow(d[2], 2) * d[7] - std::pow(d[3], 2) * d[7] +
                  std::pow(d[4], 2) * d[7] - std::pow(d[5], 2) * d[7] + 2 * d[1] * d[2] * d[8] + 2 * d[4] * d[5] * d[8];
    coeffs[160] = std::pow(d[15], 2) * d[17] + std::pow(d[16], 2) * d[17] + std::pow(d[17], 3);
    coeffs[161] = 2 * d[15] * d[17] * d[24] + 2 * d[16] * d[17] * d[25] + std::pow(d[15], 2) * d[26] +
                  std::pow(d[16], 2) * d[26] + 3 * std::pow(d[17], 2) * d[26];
    coeffs[162] = d[17] * std::pow(d[24], 2) + d[17] * std::pow(d[25], 2) + 2 * d[15] * d[24] * d[26] +
                  2 * d[16] * d[25] * d[26] + 3 * d[17] * std::pow(d[26], 2);
    coeffs[163] = std::pow(d[24], 2) * d[26] + std::pow(d[25], 2) * d[26] + std::pow(d[26], 3);
    coeffs[164] = 2 * d[9] * d[11] * d[15] + 2 * d[12] * d[14] * d[15] + 2 * d[10] * d[11] * d[16] +
                  2 * d[13] * d[14] * d[16] - std::pow(d[9], 2) * d[17] - std::pow(d[10], 2) * d[17] +
                  std::pow(d[11], 2) * d[17] - std::pow(d[12], 2) * d[17] - std::pow(d[13], 2) * d[17] +
                  std::pow(d[14], 2) * d[17];
    coeffs[165] = 2 * d[11] * d[15] * d[18] - 2 * d[9] * d[17] * d[18] + 2 * d[11] * d[16] * d[19] -
                  2 * d[10] * d[17] * d[19] + 2 * d[9] * d[15] * d[20] + 2 * d[10] * d[16] * d[20] +
                  2 * d[11] * d[17] * d[20] + 2 * d[14] * d[15] * d[21] - 2 * d[12] * d[17] * d[21] +
                  2 * d[14] * d[16] * d[22] - 2 * d[13] * d[17] * d[22] + 2 * d[12] * d[15] * d[23] +
                  2 * d[13] * d[16] * d[23] + 2 * d[14] * d[17] * d[23] + 2 * d[9] * d[11] * d[24] +
                  2 * d[12] * d[14] * d[24] + 2 * d[10] * d[11] * d[25] + 2 * d[13] * d[14] * d[25] -
                  std::pow(d[9], 2) * d[26] - std::pow(d[10], 2) * d[26] + std::pow(d[11], 2) * d[26] -
                  std::pow(d[12], 2) * d[26] - std::pow(d[13], 2) * d[26] + std::pow(d[14], 2) * d[26];
    coeffs[166] = -d[17] * std::pow(d[18], 2) - d[17] * std::pow(d[19], 2) + 2 * d[15] * d[18] * d[20] +
                  2 * d[16] * d[19] * d[20] + d[17] * std::pow(d[20], 2) - d[17] * std::pow(d[21], 2) -
                  d[17] * std::pow(d[22], 2) + 2 * d[15] * d[21] * d[23] + 2 * d[16] * d[22] * d[23] +
                  d[17] * std::pow(d[23], 2) + 2 * d[11] * d[18] * d[24] + 2 * d[9] * d[20] * d[24] +
                  2 * d[14] * d[21] * d[24] + 2 * d[12] * d[23] * d[24] + 2 * d[11] * d[19] * d[25] +
                  2 * d[10] * d[20] * d[25] + 2 * d[14] * d[22] * d[25] + 2 * d[13] * d[23] * d[25] -
                  2 * d[9] * d[18] * d[26] - 2 * d[10] * d[19] * d[26] + 2 * d[11] * d[20] * d[26] -
                  2 * d[12] * d[21] * d[26] - 2 * d[13] * d[22] * d[26] + 2 * d[14] * d[23] * d[26];
    coeffs[167] = 2 * d[18] * d[20] * d[24] + 2 * d[21] * d[23] * d[24] + 2 * d[19] * d[20] * d[25] +
                  2 * d[22] * d[23] * d[25] - std::pow(d[18], 2) * d[26] - std::pow(d[19], 2) * d[26] +
                  std::pow(d[20], 2) * d[26] - std::pow(d[21], 2) * d[26] - std::pow(d[22], 2) * d[26] +
                  std::pow(d[23], 2) * d[26];
    coeffs[168] = d[8] * std::pow(d[15], 2) + d[8] * std::pow(d[16], 2) + 2 * d[6] * d[15] * d[17] +
                  2 * d[7] * d[16] * d[17] + 3 * d[8] * std::pow(d[17], 2);
    coeffs[169] = 2 * d[8] * d[15] * d[24] + 2 * d[6] * d[17] * d[24] + 2 * d[8] * d[16] * d[25] +
                  2 * d[7] * d[17] * d[25] + 2 * d[6] * d[15] * d[26] + 2 * d[7] * d[16] * d[26] +
                  6 * d[8] * d[17] * d[26];
    coeffs[170] = d[8] * std::pow(d[24], 2) + d[8] * std::pow(d[25], 2) + 2 * d[6] * d[24] * d[26] +
                  2 * d[7] * d[25] * d[26] + 3 * d[8] * std::pow(d[26], 2);
    coeffs[171] =
        -d[8] * std::pow(d[9], 2) - d[8] * std::pow(d[10], 2) + 2 * d[6] * d[9] * d[11] + 2 * d[7] * d[10] * d[11] +
        d[8] * std::pow(d[11], 2) - d[8] * std::pow(d[12], 2) - d[8] * std::pow(d[13], 2) + 2 * d[6] * d[12] * d[14] +
        2 * d[7] * d[13] * d[14] + d[8] * std::pow(d[14], 2) + 2 * d[2] * d[9] * d[15] + 2 * d[0] * d[11] * d[15] +
        2 * d[5] * d[12] * d[15] + 2 * d[3] * d[14] * d[15] + 2 * d[2] * d[10] * d[16] + 2 * d[1] * d[11] * d[16] +
        2 * d[5] * d[13] * d[16] + 2 * d[4] * d[14] * d[16] - 2 * d[0] * d[9] * d[17] - 2 * d[1] * d[10] * d[17] +
        2 * d[2] * d[11] * d[17] - 2 * d[3] * d[12] * d[17] - 2 * d[4] * d[13] * d[17] + 2 * d[5] * d[14] * d[17];
    coeffs[172] =
        -2 * d[8] * d[9] * d[18] + 2 * d[6] * d[11] * d[18] + 2 * d[2] * d[15] * d[18] - 2 * d[0] * d[17] * d[18] -
        2 * d[8] * d[10] * d[19] + 2 * d[7] * d[11] * d[19] + 2 * d[2] * d[16] * d[19] - 2 * d[1] * d[17] * d[19] +
        2 * d[6] * d[9] * d[20] + 2 * d[7] * d[10] * d[20] + 2 * d[8] * d[11] * d[20] + 2 * d[0] * d[15] * d[20] +
        2 * d[1] * d[16] * d[20] + 2 * d[2] * d[17] * d[20] - 2 * d[8] * d[12] * d[21] + 2 * d[6] * d[14] * d[21] +
        2 * d[5] * d[15] * d[21] - 2 * d[3] * d[17] * d[21] - 2 * d[8] * d[13] * d[22] + 2 * d[7] * d[14] * d[22] +
        2 * d[5] * d[16] * d[22] - 2 * d[4] * d[17] * d[22] + 2 * d[6] * d[12] * d[23] + 2 * d[7] * d[13] * d[23] +
        2 * d[8] * d[14] * d[23] + 2 * d[3] * d[15] * d[23] + 2 * d[4] * d[16] * d[23] + 2 * d[5] * d[17] * d[23] +
        2 * d[2] * d[9] * d[24] + 2 * d[0] * d[11] * d[24] + 2 * d[5] * d[12] * d[24] + 2 * d[3] * d[14] * d[24] +
        2 * d[2] * d[10] * d[25] + 2 * d[1] * d[11] * d[25] + 2 * d[5] * d[13] * d[25] + 2 * d[4] * d[14] * d[25] -
        2 * d[0] * d[9] * d[26] - 2 * d[1] * d[10] * d[26] + 2 * d[2] * d[11] * d[26] - 2 * d[3] * d[12] * d[26] -
        2 * d[4] * d[13] * d[26] + 2 * d[5] * d[14] * d[26];
    coeffs[173] =
        -d[8] * std::pow(d[18], 2) - d[8] * std::pow(d[19], 2) + 2 * d[6] * d[18] * d[20] + 2 * d[7] * d[19] * d[20] +
        d[8] * std::pow(d[20], 2) - d[8] * std::pow(d[21], 2) - d[8] * std::pow(d[22], 2) + 2 * d[6] * d[21] * d[23] +
        2 * d[7] * d[22] * d[23] + d[8] * std::pow(d[23], 2) + 2 * d[2] * d[18] * d[24] + 2 * d[0] * d[20] * d[24] +
        2 * d[5] * d[21] * d[24] + 2 * d[3] * d[23] * d[24] + 2 * d[2] * d[19] * d[25] + 2 * d[1] * d[20] * d[25] +
        2 * d[5] * d[22] * d[25] + 2 * d[4] * d[23] * d[25] - 2 * d[0] * d[18] * d[26] - 2 * d[1] * d[19] * d[26] +
        2 * d[2] * d[20] * d[26] - 2 * d[3] * d[21] * d[26] - 2 * d[4] * d[22] * d[26] + 2 * d[5] * d[23] * d[26];
    coeffs[174] = 2 * d[6] * d[8] * d[15] + 2 * d[7] * d[8] * d[16] + std::pow(d[6], 2) * d[17] +
                  std::pow(d[7], 2) * d[17] + 3 * std::pow(d[8], 2) * d[17];
    coeffs[175] = 2 * d[6] * d[8] * d[24] + 2 * d[7] * d[8] * d[25] + std::pow(d[6], 2) * d[26] +
                  std::pow(d[7], 2) * d[26] + 3 * std::pow(d[8], 2) * d[26];
    coeffs[176] =
        2 * d[2] * d[6] * d[9] - 2 * d[0] * d[8] * d[9] + 2 * d[2] * d[7] * d[10] - 2 * d[1] * d[8] * d[10] +
        2 * d[0] * d[6] * d[11] + 2 * d[1] * d[7] * d[11] + 2 * d[2] * d[8] * d[11] + 2 * d[5] * d[6] * d[12] -
        2 * d[3] * d[8] * d[12] + 2 * d[5] * d[7] * d[13] - 2 * d[4] * d[8] * d[13] + 2 * d[3] * d[6] * d[14] +
        2 * d[4] * d[7] * d[14] + 2 * d[5] * d[8] * d[14] + 2 * d[0] * d[2] * d[15] + 2 * d[3] * d[5] * d[15] +
        2 * d[1] * d[2] * d[16] + 2 * d[4] * d[5] * d[16] - std::pow(d[0], 2) * d[17] - std::pow(d[1], 2) * d[17] +
        std::pow(d[2], 2) * d[17] - std::pow(d[3], 2) * d[17] - std::pow(d[4], 2) * d[17] + std::pow(d[5], 2) * d[17];
    coeffs[177] =
        2 * d[2] * d[6] * d[18] - 2 * d[0] * d[8] * d[18] + 2 * d[2] * d[7] * d[19] - 2 * d[1] * d[8] * d[19] +
        2 * d[0] * d[6] * d[20] + 2 * d[1] * d[7] * d[20] + 2 * d[2] * d[8] * d[20] + 2 * d[5] * d[6] * d[21] -
        2 * d[3] * d[8] * d[21] + 2 * d[5] * d[7] * d[22] - 2 * d[4] * d[8] * d[22] + 2 * d[3] * d[6] * d[23] +
        2 * d[4] * d[7] * d[23] + 2 * d[5] * d[8] * d[23] + 2 * d[0] * d[2] * d[24] + 2 * d[3] * d[5] * d[24] +
        2 * d[1] * d[2] * d[25] + 2 * d[4] * d[5] * d[25] - std::pow(d[0], 2) * d[26] - std::pow(d[1], 2) * d[26] +
        std::pow(d[2], 2) * d[26] - std::pow(d[3], 2) * d[26] - std::pow(d[4], 2) * d[26] + std::pow(d[5], 2) * d[26];
    coeffs[178] = std::pow(d[6], 2) * d[8] + std::pow(d[7], 2) * d[8] + std::pow(d[8], 3);
    coeffs[179] = 2 * d[0] * d[2] * d[6] + 2 * d[3] * d[5] * d[6] + 2 * d[1] * d[2] * d[7] + 2 * d[4] * d[5] * d[7] -
                  std::pow(d[0], 2) * d[8] - std::pow(d[1], 2) * d[8] + std::pow(d[2], 2) * d[8] -
                  std::pow(d[3], 2) * d[8] - std::pow(d[4], 2) * d[8] + std::pow(d[5], 2) * d[8];
    coeffs[180] = -d[11] * d[13] * d[15] + d[10] * d[14] * d[15] + d[11] * d[12] * d[16] - d[9] * d[14] * d[16] -
                  d[10] * d[12] * d[17] + d[9] * d[13] * d[17];
    coeffs[181] = -d[14] * d[16] * d[18] + d[13] * d[17] * d[18] + d[14] * d[15] * d[19] - d[12] * d[17] * d[19] -
                  d[13] * d[15] * d[20] + d[12] * d[16] * d[20] + d[11] * d[16] * d[21] - d[10] * d[17] * d[21] -
                  d[11] * d[15] * d[22] + d[9] * d[17] * d[22] + d[10] * d[15] * d[23] - d[9] * d[16] * d[23] -
                  d[11] * d[13] * d[24] + d[10] * d[14] * d[24] + d[11] * d[12] * d[25] - d[9] * d[14] * d[25] -
                  d[10] * d[12] * d[26] + d[9] * d[13] * d[26];
    coeffs[182] = -d[17] * d[19] * d[21] + d[16] * d[20] * d[21] + d[17] * d[18] * d[22] - d[15] * d[20] * d[22] -
                  d[16] * d[18] * d[23] + d[15] * d[19] * d[23] + d[14] * d[19] * d[24] - d[13] * d[20] * d[24] -
                  d[11] * d[22] * d[24] + d[10] * d[23] * d[24] - d[14] * d[18] * d[25] + d[12] * d[20] * d[25] +
                  d[11] * d[21] * d[25] - d[9] * d[23] * d[25] + d[13] * d[18] * d[26] - d[12] * d[19] * d[26] -
                  d[10] * d[21] * d[26] + d[9] * d[22] * d[26];
    coeffs[183] = -d[20] * d[22] * d[24] + d[19] * d[23] * d[24] + d[20] * d[21] * d[25] - d[18] * d[23] * d[25] -
                  d[19] * d[21] * d[26] + d[18] * d[22] * d[26];
    coeffs[184] = -d[8] * d[10] * d[12] + d[7] * d[11] * d[12] + d[8] * d[9] * d[13] - d[6] * d[11] * d[13] -
                  d[7] * d[9] * d[14] + d[6] * d[10] * d[14] + d[5] * d[10] * d[15] - d[4] * d[11] * d[15] -
                  d[2] * d[13] * d[15] + d[1] * d[14] * d[15] - d[5] * d[9] * d[16] + d[3] * d[11] * d[16] +
                  d[2] * d[12] * d[16] - d[0] * d[14] * d[16] + d[4] * d[9] * d[17] - d[3] * d[10] * d[17] -
                  d[1] * d[12] * d[17] + d[0] * d[13] * d[17];
    coeffs[185] = d[8] * d[13] * d[18] - d[7] * d[14] * d[18] - d[5] * d[16] * d[18] + d[4] * d[17] * d[18] -
                  d[8] * d[12] * d[19] + d[6] * d[14] * d[19] + d[5] * d[15] * d[19] - d[3] * d[17] * d[19] +
                  d[7] * d[12] * d[20] - d[6] * d[13] * d[20] - d[4] * d[15] * d[20] + d[3] * d[16] * d[20] -
                  d[8] * d[10] * d[21] + d[7] * d[11] * d[21] + d[2] * d[16] * d[21] - d[1] * d[17] * d[21] +
                  d[8] * d[9] * d[22] - d[6] * d[11] * d[22] - d[2] * d[15] * d[22] + d[0] * d[17] * d[22] -
                  d[7] * d[9] * d[23] + d[6] * d[10] * d[23] + d[1] * d[15] * d[23] - d[0] * d[16] * d[23] +
                  d[5] * d[10] * d[24] - d[4] * d[11] * d[24] - d[2] * d[13] * d[24] + d[1] * d[14] * d[24] -
                  d[5] * d[9] * d[25] + d[3] * d[11] * d[25] + d[2] * d[12] * d[25] - d[0] * d[14] * d[25] +
                  d[4] * d[9] * d[26] - d[3] * d[10] * d[26] - d[1] * d[12] * d[26] + d[0] * d[13] * d[26];
    coeffs[186] = -d[8] * d[19] * d[21] + d[7] * d[20] * d[21] + d[8] * d[18] * d[22] - d[6] * d[20] * d[22] -
                  d[7] * d[18] * d[23] + d[6] * d[19] * d[23] + d[5] * d[19] * d[24] - d[4] * d[20] * d[24] -
                  d[2] * d[22] * d[24] + d[1] * d[23] * d[24] - d[5] * d[18] * d[25] + d[3] * d[20] * d[25] +
                  d[2] * d[21] * d[25] - d[0] * d[23] * d[25] + d[4] * d[18] * d[26] - d[3] * d[19] * d[26] -
                  d[1] * d[21] * d[26] + d[0] * d[22] * d[26];
    coeffs[187] = -d[5] * d[7] * d[9] + d[4] * d[8] * d[9] + d[5] * d[6] * d[10] - d[3] * d[8] * d[10] -
                  d[4] * d[6] * d[11] + d[3] * d[7] * d[11] + d[2] * d[7] * d[12] - d[1] * d[8] * d[12] -
                  d[2] * d[6] * d[13] + d[0] * d[8] * d[13] + d[1] * d[6] * d[14] - d[0] * d[7] * d[14] -
                  d[2] * d[4] * d[15] + d[1] * d[5] * d[15] + d[2] * d[3] * d[16] - d[0] * d[5] * d[16] -
                  d[1] * d[3] * d[17] + d[0] * d[4] * d[17];
    coeffs[188] = -d[5] * d[7] * d[18] + d[4] * d[8] * d[18] + d[5] * d[6] * d[19] - d[3] * d[8] * d[19] -
                  d[4] * d[6] * d[20] + d[3] * d[7] * d[20] + d[2] * d[7] * d[21] - d[1] * d[8] * d[21] -
                  d[2] * d[6] * d[22] + d[0] * d[8] * d[22] + d[1] * d[6] * d[23] - d[0] * d[7] * d[23] -
                  d[2] * d[4] * d[24] + d[1] * d[5] * d[24] + d[2] * d[3] * d[25] - d[0] * d[5] * d[25] -
                  d[1] * d[3] * d[26] + d[0] * d[4] * d[26];
    coeffs[189] = -d[2] * d[4] * d[6] + d[1] * d[5] * d[6] + d[2] * d[3] * d[7] - d[0] * d[5] * d[7] -
                  d[1] * d[3] * d[8] + d[0] * d[4] * d[8];

    // Setup elimination template
    static const int coeffs0_ind[] = {
        0,   20,  40,  60,  80,  100, 120, 140, 160, 180, 1,   21,  41,  61,  81,  101, 121, 141, 161, 181, 2,
        22,  42,  62,  82,  102, 122, 142, 162, 182, 3,   23,  43,  63,  83,  103, 123, 143, 163, 183, 4,   24,
        44,  64,  40,  20,  60,  0,   84,  104, 124, 80,  120, 100, 144, 140, 160, 164, 180, 5,   25,  45,  65,
        41,  21,  61,  1,   85,  105, 125, 81,  121, 101, 145, 141, 161, 165, 181, 6,   26,  46,  66,  42,  22,
        62,  2,   86,  106, 126, 82,  122, 102, 146, 142, 162, 166, 182, 7,   27,  47,  67,  43,  23,  63,  3,
        87,  107, 127, 83,  123, 103, 147, 143, 163, 167, 183, 8,   28,  48,  68,  88,  108, 128, 148, 168, 184,
        9,   29,  49,  69,  89,  109, 129, 149, 169, 185, 10,  30,  50,  70,  90,  110, 130, 150, 170, 186, 44,
        24,  64,  4,   84,  124, 104, 144, 164, 180, 45,  25,  65,  5,   85,  125, 105, 145, 165, 181, 11,  31,
        51,  71,  48,  28,  68,  8,   91,  111, 131, 88,  128, 108, 151, 148, 168, 171, 184, 14,  34,  54,  74,
        94,  114, 134, 154, 174, 187, 51,  31,  71,  11,  91,  131, 111, 151, 171, 184, 46,  26,  66,  6,   86,
        126, 106, 146, 166, 182, 12,  32,  52,  72,  49,  29,  69,  9,   92,  112, 132, 89,  129, 109, 152, 149,
        169, 172, 185, 47,  27,  67,  7,   87,  127, 107, 147, 167, 183, 13,  33,  53,  73,  50,  30,  70,  10,
        93,  113, 133, 90,  130, 110, 153, 150, 170, 173, 186, 15,  35,  55,  75,  95,  115, 135, 155, 175, 188};
    static const int coeffs1_ind[] = {
        59,  39,  79,  19,  99,  139, 119, 159, 179, 189, 56, 36, 76,  16,  96,  136, 116, 156, 176, 187,
        52,  32,  72,  12,  92,  132, 112, 152, 172, 185, 16, 36, 56,  76,  54,  34,  74,  14,  96,  116,
        136, 94,  134, 114, 156, 154, 174, 176, 187, 57,  37, 77, 17,  97,  137, 117, 157, 177, 188, 53,
        33,  73,  13,  93,  133, 113, 153, 173, 186, 17,  37, 57, 77,  55,  35,  75,  15,  97,  117, 137,
        95,  135, 115, 157, 155, 175, 177, 188, 19,  39,  59, 79, 58,  38,  78,  18,  99,  119, 139, 98,
        138, 118, 159, 158, 178, 179, 189, 18,  38,  58,  78, 98, 118, 138, 158, 178, 189};
    static const int C0_ind[] = {
        0,   1,   2,   3,   8,   9,   10,  14,  17,  20,  21,  22,  23,  24,  29,  30,  31,  35,  38,  41,  42,
        43,  44,  45,  50,  51,  52,  56,  59,  62,  63,  64,  65,  66,  71,  72,  73,  77,  80,  83,  84,  85,
        86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 103, 105, 106, 107, 108,
        109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 124, 126, 127, 128, 129, 130, 131,
        132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 145, 147, 148, 149, 150, 151, 152, 153, 154,
        155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 166, 168, 169, 170, 171, 176, 177, 178, 182, 185, 188,
        189, 190, 191, 192, 197, 198, 199, 203, 206, 209, 210, 211, 212, 213, 218, 219, 220, 224, 227, 230, 235,
        236, 237, 238, 242, 243, 244, 246, 247, 249, 256, 257, 258, 259, 263, 264, 265, 267, 268, 270, 273, 274,
        275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 292, 294, 295, 296, 297,
        302, 303, 304, 308, 311, 314, 319, 320, 321, 322, 326, 327, 328, 330, 331, 333, 340, 341, 342, 343, 347,
        348, 349, 351, 352, 354, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372,
        373, 374, 376, 382, 383, 384, 385, 389, 390, 391, 393, 394, 396, 399, 400, 401, 402, 403, 404, 405, 406,
        407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 418, 420, 421, 422, 423, 428, 429, 430, 434, 437, 440};
    static const int C1_ind[] = {4,   5,   6,   7,   11,  12,  13,  15,  16,  18,  25,  26,  27,  28,  32,  33,  34,
                                 36,  37,  39,  46,  47,  48,  49,  53,  54,  55,  57,  58,  60,  63,  64,  65,  66,
                                 67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  82,  88,  89,
                                 90,  91,  95,  96,  97,  99,  100, 102, 109, 110, 111, 112, 116, 117, 118, 120, 121,
                                 123, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
                                 142, 143, 145, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
                                 161, 162, 163, 164, 166, 168, 169, 170, 171, 176, 177, 178, 182, 185, 188};

    Eigen::Matrix<double, 21, 21> C0;
    C0.setZero();
    Eigen::Matrix<double, 21, 9> C1;
    C1.setZero();
    for (int i = 0; i < 273; i++) {
        C0(C0_ind[i]) = coeffs(coeffs0_ind[i]);
    }
    for (int i = 0; i < 117; i++) {
        C1(C1_ind[i]) = coeffs(coeffs1_ind[i]);
    }

    Eigen::Matrix<double, 21, 9> C12 = C0.partialPivLu().solve(C1);

    // Setup action matrix
    Eigen::Matrix<double, 14, 9> RR;
    RR << -C12.bottomRows(5), Eigen::Matrix<double, 9, 9>::Identity(9, 9);

    static const int AM_ind[] = {9, 7, 0, 1, 10, 2, 3, 11, 4};
    Eigen::Matrix<double, 9, 9> AM;
    for (int i = 0; i < 9; i++) {
        AM.row(i) = RR.row(AM_ind[i]);
    }

    // sols.setZero();

    // Solve eigenvalue problem
    Eigen::EigenSolver<Eigen::Matrix<double, 9, 9>> es(AM);
    Eigen::ArrayXcd D = es.eigenvalues();
    Eigen::ArrayXXcd V = es.eigenvectors();
    V = (V / V.row(0).array().replicate(9, 1)).eval();

    sols.row(0) = V.row(1).array();
    sols.row(1) = D.transpose().array();
    sols.row(2) = V.row(7).array();

    int nroots = 9;

    return nroots;
}

namespace poselib {

int relpose_6pt_onefocal(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                         CameraOneFocaPoseVector *out_focal_poses) {

    // Compute nullspace to epipolar constraints
    Eigen::Matrix<double, 9, 6> epipolar_constraints;
    for (size_t i = 0; i < 6; ++i) {
        epipolar_constraints.col(i) << x1[i](0) * x2[i], x1[i](1) * x2[i], x1[i](2) * x2[i];
    }
    Eigen::Matrix<double, 9, 9> Q = epipolar_constraints.fullPivHouseholderQr().matrixQ();
    Eigen::Matrix<double, 9, 3> N = Q.rightCols(3);

    Eigen::VectorXd B(Eigen::Map<Eigen::VectorXd>(N.data(), N.cols() * N.rows()));

    Eigen::Matrix<std::complex<double>, 3, 9> sols;

    int n_sols = solver_relpose_6pt_onefocal(B, sols);

    out_focal_poses->empty();

    int n_poses = 0;

    for (int i = 0; i < n_sols; i++) {
        if (sols(2, i).real() < 1e-8 || sols.col(i).imag().norm() > 1e-8) {
            continue;
        }

        double focal = std::sqrt(1.0 / sols(2, i).real());

        // std::cout << "Focal: " << focal << ", Sol: " << sols.col(i) << ", Sol img norm: " << sols.col(i).imag().norm() << "\n"; 

        Eigen::Vector<double, 9> F_vector =
            N.col(0) + sols(0, i).real() * N.col(1) + sols(1, i).real() * N.col(2);
        F_vector.normalize();
        Eigen::Matrix3d F = Eigen::Matrix3d(F_vector.data());

        //std::cout << "Inlier epipolar dist: " << x2[0].transpose() * (F * x1[0]) << "\n";
                        
        Eigen::Matrix3d K;
        Eigen::Matrix3d K_inv;
        K << focal, 0.0, 0.0, 0.0, focal, 0.0, 0.0, 0.0, 1.0;
        K_inv << 1 / focal, 0.0, 0.0, 0.0, 1 / focal, 0.0, 0.0, 0.0, 1.0;
        
        Eigen::Matrix3d E = F * K;
        
        CameraPoseVector poses;
        motion_from_essential(E, K_inv * x1[0], x2[0], &poses);
        
        for (CameraPose pose : poses) {
            out_focal_poses->emplace_back(CameraOneFocalPose(pose, focal));
            n_poses++;
        }        
    }

    return n_poses;
}
} // namespace poselib
