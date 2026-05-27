#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define M_PI 3.14159265358979
#define ASMALLNUMBER 1E-6
#define AREALLYSMALLNUMBER 1E-11
#define idx(ilv,point,em) ((ilv)*localNPTS*3+(point)*3+em)

float SQ(float x)
{
    return ((x) * (x));
}

float SIGN(float x)
{
    return (((x) >= 0.0) ? 1.0 : -1.0);
}

void matmul(float a00, float a01, float a02, float a10, float a11,
    float a12, float a20, float a21, float a22, float x[3])
{
    // matrix multiply a (3x3) by x (3x1)
    float xp0 = a00 * x[0] + a01 * x[1] + a02 * x[2];
    float xp1 = a10 * x[0] + a11 * x[1] + a12 * x[2];
    x[2] = a20 * x[0] + a21 * x[1] + a22 * x[2];
    x[0] = xp0;
    x[1] = xp1;
}

void rotarb(float *v, float th, float *u)
{
    // rotates vector v around axis u through angle th
    float c = cos(th), s = sin(th), cm = 1 - c;
    matmul(c + SQ(u[0]) * cm, u[0] * u[1] * cm - u[2] * s, u[0] * u[2] * cm + u[1] * s,
        u[1] * u[0] * cm + u[2] * s, c + SQ(u[1]) * cm, u[1] * u[2] * cm - u[0] * s,
        u[2] * u[0] * cm - u[1] * s, u[2] * u[1] * cm + u[0] * s, c + SQ(u[2]) * cm, v);
}

float dot(float *x, float *y)
{
    // dot product
    return(x[0] * y[0] + x[1] * y[1] + x[2] * y[2]);
}

void norm(float *x, float r)
{
    // normalize 3-vector x
    float n = (float)(sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]) / r);
    x[0] /= n;
    x[1] /= n;
    x[2] /= n;
}

void cpyvec(float *x, float *y)
{
    // copy 3-vector y --> x
    x[0] = y[0];
    x[1] = y[1];
    x[2] = y[2];
}

void cross(float *x, float *y, float *result)
{
    result[0] = x[1] * y[2] - y[1] * x[2];
    result[1] = y[0] * x[2] - x[0] * y[2];
    result[2] = x[0] * y[1] - y[0] * x[1];
    norm(result, 1.0);
}

float calct1(float atp, float kmax, float kddmax, float n)
{
    if ((n >= 2.0 - ASMALLNUMBER) && (n <= 2.0 + ASMALLNUMBER))
        return(atp);
    return(atp / (pow(2.0 * kmax / kddmax / atp / atp, 1.0 / (n - 2.0)) - 1.0));
}

float krp(float tp, float kmax, float t1, float atp, float n)
{
    float A = kmax / atp / atp / pow(1 + atp / t1, n - 2.0);
    return(A * tp * tp * pow(1 + tp / t1, n - 2.0));
}

float kr(float t, float t0, float kddmax, float ms, float fov, float AT, float n)
{
    if (t < t0)
        return(0.0);
    float kmax = ms / (2 * fov);
    float t1 = calct1(AT - t0, kmax, kddmax, n);
    return(krp(t - t0, kmax, t1, AT - t0, n));
}

float krdotp(float tp, float kmax, float t1, float atp, float n)
{
    if (tp < ASMALLNUMBER)
        return(0.0);
    float A = kmax / atp / atp / pow(1 + atp / t1, n - 2.0);
    return(A * tp * pow(1 + tp / t1, n-3) * (2 + n * tp / t1));
}

float krdot(float t, float t0, float kddmax, float ms, float fov, float at, float n)
{
    if (t < t0)
        return(0.0);
    float kmax = ms / (2 * fov);
    float t1 = calct1(at - t0, kmax, kddmax, n);
    return(krdotp(t - t0, kmax, t1, at - t0, n));
}


float krddot(float t, float t0, float kddmax, float ms, float fov, float at, float n)
{
    float tp = t - t0;
    if (tp < ASMALLNUMBER)
        return(0.0);
    float kmax = ms / (2 * fov);
    float atp = at - t0;
    float t1 = calct1(at - t0, ms / (2 * fov), kddmax, n);
    float A = kmax / atp / atp / pow(1 + atp / t1, n - 2.0);
    float newway = A * pow(1 + tp / t1, n - 4) * (2 + (n - 1) * tp / t1 * (4 + n * tp / t1));
    return(newway);
}

void optimizehedgehog(int n, float *v)
{
    float *vel = (float *)malloc(3 * n * sizeof(float));
    float *dvtot = (float *)malloc(3 * n * sizeof(float));
    float dv[3];
    int i, j, k;
    float sqn = sqrt((float)n);

    for(i = 0; i < 3 * n; i++)
        vel[i] = 0.0;
    for(i = 0; i < 10000; i++)
    {
        float maxnorm = 0.0;
        for(j = 0; j < n; j++)
        {
            int jidx = j * 3;
            // calculate transverse force on this one
            for(k = j + 1; k < n; k++)
            {   
                int kidx = k * 3;
                dv[0] = v[jidx] - v[kidx];
                dv[1] = v[jidx + 1] - v[kidx + 1];
                dv[2] = v[jidx + 2] - v[kidx + 2];
                float norm = dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2];
                float denom = norm * sqn * 3;
                float jdot = dv[0] * v[jidx] + dv[1] * v[jidx + 1] + dv[2] * v[jidx + 2];
                vel[jidx] += (dv[0] - jdot * v[j * 3]) / denom;
                vel[jidx + 1] += (dv[1] - jdot * v[jidx + 1]) / denom;
                vel[jidx + 2] += (dv[2] - jdot * v[jidx + 2]) / denom;
                float kdot = dv[0] * v[kidx] + dv[1] * v[kidx + 1] + dv[2] * v[kidx + 2];
                vel[kidx] -= (dv[0] - kdot * v[kidx]) / denom;
                vel[kidx + 1] -= (dv[1] - kdot * v[kidx + 1]) / denom;
                vel[kidx + 2] -= (dv[2] - kdot * v[kidx + 2]) / denom;
            }
        }
        for(j = 0; j < n; j++)
        {
            int jidx = j * 3;
            v[jidx] += vel[jidx];
            v[jidx + 1] += vel[jidx + 1];
            v[jidx + 2] += vel[jidx + 2];
            float norm = sqrt(v[jidx] * v[jidx] + v[jidx + 1] * v[jidx + 1] + v[jidx + 2] * v[jidx + 2]);
            v[jidx] /= norm;
            v[jidx + 1] /= norm;
            v[jidx + 2] /= norm;
            vel[jidx] *= 0.9;
            vel[jidx + 1] *= 0.9;
            vel[jidx + 2] *= 0.9;
            float velnorm = vel[jidx] * vel[jidx] + vel[jidx + 1] * vel[jidx + 1] + vel[jidx + 2] * vel[jidx + 2];
            if(velnorm > maxnorm)
                maxnorm = velnorm;
        }
        if(maxnorm < AREALLYSMALLNUMBER)
            return;
    }
}

void calchedgehog(int n, int optimize, float *target)
{
    float phi = M_PI * (sqrt(5.0) - 1.0);
    for(int i = 0; i < n; i++)
    {
        int idx = 3 * i;
        float theta = phi * i;
        target[idx] = 1.0 - (i / (float)(n - 1)) * 2.0;
        float r = sqrt(1.0 - target[idx] * target[idx]);
        target[idx + 1] = cos(theta) * r;
        target[idx + 2] = sin(theta) * r;
        norm(&target[idx], 1.0);
    }
    for(int i = 0; i < n; i++)
        printf("%f %f %f\n", target[3*i], target[3*i+1], target[3*i+2]);
    if(optimize)
        optimizehedgehog(n, target);
     for(int i = 0; i < n; i++)
        printf("%f %f %f\n", target[3*i], target[3*i+1], target[3*i+2]);
}

// Global basis vector sets for interleaves (initv) and rotations (reprot)
float *initv = (float *)0, *reprot = (float *)0;


void makegrads(float *gx, float *gy, float *gz, float at, float fov, float t0, float ms, float gam, float dt, float n, int NI, int NPTS, int NREPS, int optimize, int irep)
{
    static int localNI = -1;
    static int localNPTS = -1;
    static int localNREPS = -1;
    static int localoptimize = -1;
    static float *k = (float *)0, *g = (float *)0;

    if((NI != localNI) || (NPTS != localNPTS) || (NREPS != localNREPS) || (optimize != localoptimize))
    {
        if(k != (float *)0)
            free((void *)k);
        if(g != (float *)0)
            free((void *)g);
        if((initv != (float *)0) && (NI != localNI))
            free((void *)initv);
        if((reprot != (float *)0) && (NREPS != localNREPS))
            free((void *)initv);
        k = (float *)malloc(NI * NPTS * 3 * sizeof(float));
        g = (float *)malloc(NI * NPTS * 3 * sizeof(float));
        if((NI != localNI) || (optimize != localoptimize))
        {
            initv = (float *)malloc(NI * 3 * sizeof(float));
            calchedgehog(NI, optimize, initv);
        }
        if((NREPS != localNREPS) || (optimize != localoptimize))
        {
            reprot = (float *)malloc(NREPS * 3 * sizeof(float));
            calchedgehog(NREPS, optimize, reprot);
        }
        localNI = NI;
        localNPTS = NPTS;
        localNREPS = NREPS;
        localoptimize = optimize;
    }
    const float MAXS = 150; // T/m/s
    const float MAXG = 0.04; // T/m

    // fill in gx, gy, gz so that it goes out to maxk while no gradient
    // ever exceeds MAXS
    float gr = (1 + sqrt(5.0)) / 2;
    float rotvec[3] = { 0, 0, 1 };
    float xhat[3] = { 1, 0, 0 };

    for (int ir = 0, firstime = 1; ir < NPTS; ir++)
    {
        float t = ir * dt;
        float thisk = kr(t, t0, MAXS * gam, ms, fov, at, n);
        if (thisk < ASMALLNUMBER)
        {
           for (int j = 0; j < NI; j++)
               for (int m = 0; m < 3; m++)
                   g[idx(j,ir,m)] = 0.0;
           continue; 
        }
        float thiskdot = krdot(t, t0, MAXS * gam, ms, fov, at, n);
        float thiskddot = krddot(t, t0, MAXS * gam, ms, fov, at, n);
        float wG = sqrt(SQ(MAXG * gam)  - SQ(thiskdot)) / thisk;
        float wS = sqrt(sqrt(2 * SQ(MAXS * gam * thisk) + SQ(SQ(thiskdot)) - 2 * thisk * SQ(thiskdot) * thiskddot - \
            SQ(thisk * thiskdot)) + thisk * thiskddot - SQ(thiskdot)) / sqrt(2.00) / thisk;
        float w = (wS < wG) ? wS: wG;
	printf("%f\n", w);
        // rotate rotvec around xhat at the optimal rate
        rotarb(rotvec, w * dt, xhat);
        float rotangle = w * dt;
        for (int j = 0; j < NI; j++)
        {
            if (firstime)
                cpyvec(&k[idx(j,ir,0)], &initv[3 * j]);
            else
                cpyvec(&k[idx(j,ir,0)], &k[idx(j,ir-1,0)]);
            norm(&k[idx(j,ir,0)], thisk);
            // rotate all rays around rotvec at the optimal rate
            rotarb(&k[idx(j,ir,0)], rotangle, rotvec);
            for (int m = 0; m < 3; m++)
                g[idx(j,ir,m)] = (ir == 0)? 0.0: (k[idx(j,ir,m)] - k[idx(j,ir-1,m)]) / dt / gam;
        }
        firstime = 0;
    }
    // ramp down to g=0
    int ir = NPTS - 2;
    for (int j = 0; j < NI; j++)
        for (; sqrt(dot(&g[idx(j,ir,0)], &g[idx(j,ir,0)])) > (NPTS - 1 - ir) * dt * MAXS; ir--)
            ;
    for (int irp = ir + 1; irp < NPTS; irp++)
        for (int j = 0; j < NI; j++)
            for (int m = 0; m < 3; m++)
                g[idx(j,irp,m)] = g[idx(j,ir,m)] * (NPTS - 1 - irp) / (NPTS - 1 - ir);

    // rotate by 90 deg around NREPS rays for each unique iteration
    for (int ir = 0; ir < NPTS; ir++)
    {
        for (int j = 0; j < NI; j++)
        {
            rotarb(&g[idx(j,ir,0)], M_PI / 2, &reprot[3 * (irep % NREPS)]);
            rotarb(&k[idx(j,ir,0)], M_PI / 2, &reprot[3 * (irep % NREPS)]);
        }
    }
    // pack gradient into 1D arrays
    for (int ir = 0; ir < NPTS; ir++)
    {
        for (int j = 0; j < NI; j++)
        {
            gx[ir + j * localNPTS] = g[idx(j,ir,0)];
            gy[ir + j * localNPTS] = g[idx(j,ir,1)];
            gz[ir + j * localNPTS] = g[idx(j,ir,2)];
        }
    }    
}

int main()
{
    int NI = 26;
    int NPTS = 512;
    int NREPS = 32;
    float n = 2.0, at = NPTS * 1.0E-5;
    float gamma = 11.7889E+6;
    float fov = 350.0 / 1000.0;
    float ms = 80.0;
    //int intro = 4;                    // pts with 0 gradient at the beginning
    //int outro = 35 - (int)(fov * 15); // pts to return to 0 gradient at the end
    float kmax = ms / (2 * fov);
    static float *gx = (float *)0, *gy = (float *)0, *gz = (float *)0;

    if(gx == (float *)0)
    {
        gx = (float *)malloc(NPTS*NI*NREPS*sizeof(float));
        gy = (float *)malloc(NPTS*NI*NREPS*sizeof(float));
        gz = (float *)malloc(NPTS*NI*NREPS*sizeof(float));
        for(int p = 0; p < NPTS*NI*NREPS; p++)
           gx[p] = gy[p] = gz[p] = 99999.0;
    }

    // make gradients with unoptimized (Fibbonaci) basis sets
    for(int irep = 0; irep < NREPS; irep++)
    {
        int off = NI*NPTS*irep;
        makegrads(gx+off, gy+off, gz+off, at, fov, 4*1E-5, ms, gamma, 1.0E-5, n, NI, NPTS, NREPS, 0, irep);
    }

    // dump unoptomized basis sets
    FILE *f = fopen("ilvbasis_preopt.txt", "w");
    for(int j = 0; j < NI; j++)
        fprintf(f, "%f %f %f\n", initv[3*j], initv[3*j+1], initv[3*j+2]);
    fclose(f);

    // remake gradients with optimized (Thompson) basis sets
    for(int irep = 0; irep < NREPS; irep++)
    {
        int off = NI*NPTS*irep;
        makegrads(gx+off, gy+off, gz+off, at, fov, 4*1E-5, ms, gamma, 1.0E-5, n, NI, NPTS, NREPS, 1, irep);
    }

    // dump optimized basis sets
    f = fopen("ilvbasis_postopt.txt", "w");
    for(int j = 0; j < NI; j++)
        fprintf(f, "%f %f %f\n", initv[3*j], initv[3*j+1], initv[3*j+2]);
    fclose(f);

    // dump trajectory from optimized basis sets
    f = fopen("fancytraj.txt", "w");
    fprintf(f, "%d %d %d\n", NI, NREPS, NPTS);
    for(int i = 0; i < NPTS*NI*NREPS; i++)
        fprintf(f, "%f %f %f\n", gx[i], gy[i], gz[i]);
    fclose(f);
    free((void *)gx);
    free((void *)gy);
    free((void *)gz);
    return(0);
}
