/*
   utils.c - F.J. Estrada, Dec. 9, 2010
   Last updated: Aug. 12, 2014  -  F.J.E.

   Utilities for the ray tracer. You will need to complete
   some of the functions in this file. Look for the sections
   marked "TO DO". Be sure to read the rest of the file and
   understand how the entire code works.
*/

#include "utils.h"

// A useful 4x4 identity matrix which can be used at any point to
// initialize or reset object transformations
double eye4x4[4][4]={{1.0, 0.0, 0.0, 0.0},
                    {0.0, 1.0, 0.0, 0.0},
                    {0.0, 0.0, 1.0, 0.0},
                    {0.0, 0.0, 0.0, 1.0}};

/////////////////////////////////////////////
// Primitive data structure section
/////////////////////////////////////////////
struct point3D *newPoint(double px, double py, double pz)
{
 // Allocate a new point structure, initialize it to
 // the specified coordinates, and return a pointer
 // to it.

 struct point3D *pt=(struct point3D *)calloc(1,sizeof(struct point3D));
 if (!pt) fprintf(stderr,"Out of memory allocating point structure!\n");
 else
 {
  pt->px=px;
  pt->py=py;
  pt->pz=pz;
  pt->pw=1.0;
 }
 return(pt);
}

struct pointLS *newPLS(struct point3D *p0, double r, double g, double b)
{
 // Allocate a new point light sourse structure. Initialize the light
 // source to the specified RGB colour
 // Note that this is a point light source in that it is a single point
 // in space, if you also want a uniform direction for light over the
 // scene (a so-called directional light) you need to place the
 // light source really far away.

 struct pointLS *ls=(struct pointLS *)calloc(1,sizeof(struct pointLS));
 if (!ls) fprintf(stderr,"Out of memory allocating light source!\n");
 else
 {
  memcpy(&ls->p0,p0,sizeof(struct point3D));  // Copy light source location

  ls->col.R=r;          // Store light source colour and
  ls->col.G=g;          // intensity
  ls->col.B=b;
 }
 return(ls);
}

/////////////////////////////////////////////
// Ray and normal transforms
/////////////////////////////////////////////
inline void rayTransform(struct ray3D *ray_orig, struct ray3D *ray_transformed, struct object3D *obj)
{
 // Transforms a ray using the inverse transform for the specified object. This is so that we can
 // use the intersection test for the canonical object. Note that this has to be done carefully!

 ///////////////////////////////////////////
 // TO DO: Complete this function
 ///////////////////////////////////////////

  // A^-1(p0+d*\lambda) = A^-1*p0 + A^-1*d*\lambda
  memcpy(ray_transformed,ray_orig,sizeof(struct ray3D));
  // not let homogeneous coordinate w affect the transformation
  ray_transformed->d.pw=0;
  matVecMult(obj->Tinv,&ray_transformed->p0);
  matVecMult(obj->Tinv,&ray_transformed->d);
  ray_transformed->d.pw=1;  // reset homogeneous to 1
}

inline void normalTransform(struct point3D *n_orig, struct point3D *n_transformed, struct object3D *obj)
{
 // Computes the normal at an affinely transformed point given the original normal and the
 // object's inverse transformation. From the notes:
 // n_transformed=A^-T*n normalized.

 ///////////////////////////////////////////
 // TO DO: Complete this function
 ///////////////////////////////////////////

  // Obtain A^-1 transpose = (A^-1)^T = A^-T
  double AT[4][4];
  for (int i=0;i<4;i++) {
    for (int j=0;j<4;j++) {
      AT[i][j] = obj->Tinv[j][i];
    }
  }

  // (A^-T * n) / ||A^-T * n||
  memcpy(n_transformed,n_orig,sizeof(struct point3D));
  n_transformed->pw=0;  // again, not let pw affect the transform
  matVecMult(AT,n_transformed);
  n_transformed->pw=1;  // reset pw 
  normalize(n_transformed); // self-explanatory...
}

/////////////////////////////////////////////
// Object management section
/////////////////////////////////////////////
void insertObject(struct object3D *o, struct object3D **list)
{
 if (o==NULL) return;
 // Inserts an object into the object list.
 if (*(list)==NULL)
 {
  *(list)=o;
  (*(list))->next=NULL;
 }
 else
 {
  o->next=(*(list))->next;
  (*(list))->next=o;
 }
}

struct object3D *newPlane(double ra, double rd, double rs, double rg, double r, double g, double b, double alpha, double r_index, double shiny)
{
 // Intialize a new plane with the specified parameters:
 // ra, rd, rs, rg - Albedos for the components of the Phong model
 // r, g, b, - Colour for this plane
 // alpha - Transparency, must be set to 1 unless you are doing refraction
 // r_index - Refraction index if you are doing refraction.
 // shiny - Exponent for the specular component of the Phong model
 //
 // The plane is defined by the following vertices (CCW)
 // (1,1,0), (-1,1,0), (-1,-1,0), (1,-1,0)
 // With normal vector (0,0,1) (i.e. parallel to the XY plane)

 struct object3D *plane=(struct object3D *)calloc(1,sizeof(struct object3D));

 if (!plane) fprintf(stderr,"Unable to allocate new plane, out of memory!\n");
 else
 {
  plane->alb.ra=ra;
  plane->alb.rd=rd;
  plane->alb.rs=rs;
  plane->alb.rg=rg;
  plane->col.R=r;
  plane->col.G=g;
  plane->col.B=b;
  plane->alpha=alpha;
  plane->r_index=r_index;
  plane->shinyness=shiny;
  plane->intersect=&planeIntersect;
  plane->surfaceCoords=&planeCoordinates;
  plane->randomPoint=&planeSample;
  plane->texImg=NULL;
  plane->photonMap=NULL;
  plane->normalMap=NULL;
  memcpy(&plane->T[0][0],&eye4x4[0][0],16*sizeof(double));
  memcpy(&plane->Tinv[0][0],&eye4x4[0][0],16*sizeof(double));
  plane->textureMap=&texMap;
  plane->frontAndBack=1;
  plane->photonMapped=0;
  plane->normalMapped=0;
  plane->isCSG=0;
  plane->isLightSource=0;
  plane->CSGnext=NULL;
  plane->next=NULL;
}
 return(plane);
}

struct object3D *newSphere(double ra, double rd, double rs, double rg, double r, double g, double b, double alpha, double r_index, double shiny)
{
 // Intialize a new sphere with the specified parameters:
 // ra, rd, rs, rg - Albedos for the components of the Phong model
 // r, g, b, - Colour for this plane
 // alpha - Transparency, must be set to 1 unless you are doing refraction
 // r_index - Refraction index if you are doing refraction.
 // shiny -Exponent for the specular component of the Phong model
 //
 // This is assumed to represent a unit sphere centered at the origin.
 //

 struct object3D *sphere=(struct object3D *)calloc(1,sizeof(struct object3D));

 if (!sphere) fprintf(stderr,"Unable to allocate new sphere, out of memory!\n");
 else
 {
  sphere->alb.ra=ra;
  sphere->alb.rd=rd;
  sphere->alb.rs=rs;
  sphere->alb.rg=rg;
  sphere->col.R=r;
  sphere->col.G=g;
  sphere->col.B=b;
  sphere->alpha=alpha;
  sphere->r_index=r_index;
  sphere->shinyness=shiny;
  sphere->intersect=&sphereIntersect;
  sphere->surfaceCoords=&sphereCoordinates;
  sphere->randomPoint=&sphereSample;
  sphere->texImg=NULL;
  sphere->photonMap=NULL;
  sphere->normalMap=NULL;
  memcpy(&sphere->T[0][0],&eye4x4[0][0],16*sizeof(double));
  memcpy(&sphere->Tinv[0][0],&eye4x4[0][0],16*sizeof(double));
  sphere->textureMap=&texMap;
  sphere->frontAndBack=0;
  sphere->photonMapped=0;
  sphere->normalMapped=0;
  sphere->isCSG=0;
  sphere->isLightSource=0;
  sphere->CSGnext=NULL;
  sphere->next=NULL; }
 return(sphere);
}

struct object3D *newCyl(double ra, double rd, double rs, double rg, double r, double g, double b, double alpha, double R_index, double shiny)
{
 ///////////////////////////////////////////////////////////////////////////////////////
 // TO DO:
 // Complete the code to create and initialize a new cylinder object.
 ///////////////////////////////////////////////////////////////////////////////////////  

 struct object3D *cyl=(struct object3D *)calloc(1,sizeof(struct object3D));

 if (!cyl) fprintf(stderr,"Unable to allocate new cylinder, out of memory!\n");
 else
 {
  cyl->alb.ra=ra;
  cyl->alb.rd=rd;
  cyl->alb.rs=rs;
  cyl->alb.rg=rg;
  cyl->col.R=r;
  cyl->col.G=g;
  cyl->col.B=b;
  cyl->alpha=alpha;
  cyl->r_index=R_index;
  cyl->shinyness=shiny;
  cyl->intersect=&cylIntersect;
  cyl->surfaceCoords=&cylCoordinates;
  cyl->randomPoint=&cylSample;
  cyl->texImg=NULL;
  cyl->photonMap=NULL;
  cyl->normalMap=NULL;
  memcpy(&cyl->T[0][0],&eye4x4[0][0],16*sizeof(double));
  memcpy(&cyl->Tinv[0][0],&eye4x4[0][0],16*sizeof(double));
  cyl->textureMap=&texMap;
  cyl->frontAndBack=0;
  cyl->photonMapped=0;
  cyl->normalMapped=0;
  cyl->isCSG=0;
  cyl->isLightSource=0;
  cyl->CSGnext=NULL;
  cyl->next=NULL; }
 return(cyl);
}


///////////////////////////////////////////////////////////////////////////////////////
// TO DO:
//  Complete the functions that compute intersections for the canonical cube
//      and canonical sphere with a given ray. This is the most fundamental component
//      of the raytracer.
///////////////////////////////////////////////////////////////////////////////////////

void planeIntersect(struct object3D *plane, struct ray3D *ray, double *lambda, struct point3D *p, struct point3D *n, double *a, double *b)
{
 // Computes and returns the value of 'lambda' at the intersection
 // between the specified ray and the specified canonical plane.

 /////////////////////////////////
 // TO DO: Complete this function.
 /////////////////////////////////

 // The plane is defined by the following vertices (CCW)
 // (1,1,0), (-1,1,0), (-1,-1,0), (1,-1,0)
 // With normal vector (0,0,1) (i.e. parallel to the XY plane)

  ray3D ray_transformed;
  rayTransform(ray,&ray_transformed,plane); // deform ray

  //point3D *plane_n = newPoint(0,0,1);  // normal of the plane
  point3D plane_n;
  plane_n.px = 0;
  plane_n.py = 0;
  plane_n.pz = 1;
  plane_n.pw = 1;
  // 0 = (p0+\lambda*d).n
  // 0 = p0.n + \lambda*d.n
  // -p0.n = \lambda*d.n
  // lambda = -p0.n/d.n
  *lambda = - dot(&ray_transformed.p0,&plane_n)/dot(&ray_transformed.d,&plane_n);

  // Check if point at lambda is in the plane
  rayPosition(&ray_transformed,*lambda,p);  // components of point p should be in [-1,1]
  if (p->px < -1 || p->px > 1 || p->py < -1 || p->py > 1) {
    // no intersection
    *lambda = -1;
    return;
  } else {
    // compute texture coordinates
    *a = (p->px + 1) / 2;
    *b = (p->py + 1) / 2;
    rayPosition(ray,*lambda,p);

    // find normal at the intersection point
    normalTransform(&plane_n,n,plane);

    // test if the normal vector is pointing toward or away from the ray
    // using a vector from the intersection point to the ray
    point3D p_to_ray;
    p_to_ray.px = ray->p0.px - p->px;
    p_to_ray.py = ray->p0.py - p->py;
    p_to_ray.pz = ray->p0.pz - p->pz;
    p_to_ray.pw = 1;
    if (dot(n,&p_to_ray) < 0) {
      // if so, we want the normal to point to the opposite direction
      n->px = -n->px;
      n->py = -n->py;
      n->pz = -n->pz;
    }

    if (plane->normalMap) {
      double R, G, B;
      bilinear_interpolation(plane->normalMap, *a, *b, &R, &G, &B);
      n->px = (2 * R) - 1;
      n->py = (2 * G) - 1;
      n->pz = (2 * B) - 1;
    }
    normalize(n);
  }
}

void sphereIntersect(struct object3D *sphere, struct ray3D *ray, double *lambda, struct point3D *p, struct point3D *n, double *a, double *b)
{
 // Computes and returns the value of 'lambda' at the intersection
 // between the specified ray and the specified canonical sphere.

 /////////////////////////////////
 // TO DO: Complete this function.
 /////////////////////////////////

  ray3D ray_transformed;
  rayTransform(ray,&ray_transformed,sphere);  // deform ray

  // From lecture notes. Since sphere centers at origin, vector a is simply p0
  double A = dot(&ray_transformed.d,&ray_transformed.d);
  double B = 2 * dot(&ray_transformed.p0,&ray_transformed.d);
  double C = dot(&ray_transformed.p0,&ray_transformed.p0) - 1;
  double D = B*B - 4*A*C;

  if (D < 0) {
    *lambda = -1;
    return;
  }

  // Test the two possible intersection points
  double lambda1 = (-B + sqrt(D)) / (2*A);
  double lambda2 = (-B - sqrt(D)) / (2*A);


  // Find the smallest since smallest lambda => closest intersection point
  if (D == 0) {
    *lambda = lambda1;
  } else {
    *lambda = (lambda1 > 0) ? lambda1 : -1;
    *lambda = ((*lambda < 0 || lambda2 < *lambda) && lambda2 > 0) ? lambda2 : *lambda;
    if (*lambda == -1) {
      return;
    }
  }

  // Retrieve the coordinates of intersection point p
  rayPosition(&ray_transformed,*lambda,p);
  // compute texture coordinates
  // From Wikipedia on "UV mapping"

  *b = .5 + (atan2(p->pz,p->px)/(2*PI));
  *a = .5 - (asin(p->py)/PI);

  // And normal n at intersection point too
  normalTransform(p,n,sphere);
  rayPosition(ray,*lambda,p);

  if (sphere->normalMap) {
    double R, G, B;
    bilinear_interpolation(sphere->normalMap, *a, *b, &R, &G, &B);
    n->px = (2 * R) - 1;
    n->py = (2 * G) - 1;
    n->pz = (2 * B) - 1;
  }
  normalize(n);
}

void cylIntersect(struct object3D *cylinder, struct ray3D *r, double *lambda, struct point3D *p, struct point3D *n, double *a, double *b)
{
 // Computes and returns the value of 'lambda' at the intersection
 // between the specified ray and the specified canonical cylinder.

 /////////////////////////////////
 // TO DO: Complete this function.
 /////////////////////////////////  

  ray3D ray_transformed;
  rayTransform(r,&ray_transformed,cylinder);  // deform ray

  // See report for source
  // (p0.x+d.x*\lambda)^2 + (p0.y+d.y*\lambda)^2 = 1 and solve for \lambda
  double A = pow(ray_transformed.d.px, 2)+pow(ray_transformed.d.py,2);
  double B = 2*((ray_transformed.p0.px*ray_transformed.d.px)+
    (ray_transformed.p0.py*ray_transformed.d.py));
  double C = pow(ray_transformed.p0.px,2)+pow(ray_transformed.p0.py,2)-1;
  double D = B*B - 4*A*C;

  if (D < 0) {
    *lambda = -1;
    return;
  }

  double lambda1 = (-B+sqrt(D)) / (2*A);
  double lambda2 = (-B-sqrt(D)) / (2*A);

  double z1 = ray_transformed.p0.pz+lambda1*ray_transformed.d.pz;
  double z2 = ray_transformed.p0.pz+lambda2*ray_transformed.d.pz;

  if (z1 < 0 || z1 > 1) lambda1 = -1;
  if (z2 < 0 || z2 > 1) lambda2 = -1;

  // Test for base and cap (i.e. point.pz = 0 or 1)
  // z-coord: 0 = p0.pz + d.pz * \lambda
  //          1 = p0.pz + d.pz * \lambda
  double lambda3 = -ray_transformed.p0.pz / ray_transformed.d.pz;
  double lambda4 = (1 - ray_transformed.p0.pz) / ray_transformed.d.pz;

  double x3 = ray_transformed.p0.px+lambda3*ray_transformed.d.px;
  double y3 = ray_transformed.p0.py+lambda3*ray_transformed.d.py;
  double x4 = ray_transformed.p0.px+lambda4*ray_transformed.d.px;
  double y4 = ray_transformed.p0.py+lambda4*ray_transformed.d.py;

  if ((pow(x3,2)+pow(y3,2))>1) lambda3 = -1;
  if ((pow(x4,2)+pow(y4,2))>1) lambda4 = -1;

  *lambda = (lambda1 > 0) ? lambda1 : -1;
  *lambda = (lambda2 > 0 && (*lambda < 0 || lambda2 < *lambda)) ? lambda2 : *lambda; 
  *lambda = (lambda3 > 0 && (*lambda < 0 || lambda3 < *lambda)) ? lambda3 : *lambda; 
  *lambda = (lambda4 > 0 && (*lambda < 0 || lambda4 < *lambda)) ? lambda4 : *lambda; 

  if (*lambda == -1) return;
  // Retrieve coordinates of intersection point p
  rayPosition(&ray_transformed,*lambda,p);
    // Compute texture coordinates
  if (p->pz == 0 || p->pz == 1) {
    // Treat caps as plane
    *a = (p->px + 1) / 2;
    *b = (p->py + 1) / 2;
  } else {
    *a = acos(p->px) / (2 * PI);
    *b = p->pz;
  }
  // And normal at intersection point too
  normalTransform(p,n,cylinder);
  rayPosition(r,*lambda,p);

  if (cylinder->normalMap) {
    double R, G, B;
    bilinear_interpolation(cylinder->normalMap, *a, *b, &R, &G, &B);
    n->px = (2 * R) - 1;
    n->py = (2 * G) - 1;
    n->pz = (2 * B) - 1;
  }
  normalize(n);
}

/////////////////////////////////////////////////////////////////
// Surface coordinates & random sampling on object surfaces
/////////////////////////////////////////////////////////////////
void planeCoordinates(struct object3D *plane, double a, double b, double *x, double *y, double *z)
{
 // Return in (x,y,z) the coordinates of a point on the plane given by the 2 parameters a,b in [0,1].
 // 'a' controls displacement from the left side of the plane, 'b' controls displacement from the
 // bottom of the plane.
 
 /////////////////////////////////
 // TO DO: Complete this function.
 /////////////////////////////////
  point3D p;
  p.px = (2 * a) - 1;
  p.py = (2 * b) - 1;
  p.pz = 0;
  p.pw = 1.0;
  matVecMult(plane->T,&p);

  // Reset from homogeneous coordinates
  p.px /= p.pw;
  p.py /= p.pw;
  p.pz /= p.pw;

  *x = p.px;
  *y = p.py;
  *z = p.pz;
}

void sphereCoordinates(struct object3D *sphere, double a, double b, double *x, double *y, double *z)
{
 // Return in (x,y,z) the coordinates of a point on the plane given by the 2 parameters a,b.
 // 'a' in [0, 2*PI] corresponds to the spherical coordinate theta
 // 'b' in [-PI/2, PI/2] corresponds to the spherical coordinate phi

 /////////////////////////////////
 // TO DO: Complete this function.
 ///////////////////////////////// 

  // From Wikipedia
  // x = sin(theta)*cos(phi)
  // y = sin(theta)*sin(phi)
  // z = cos(theta)

  point3D p;
  p.px = sin(a)*cos(b);
  p.py = sin(a)*sin(b);
  p.pz = cos(a);
  p.pw = 1;
  matVecMult(sphere->T,&p);

  // Reset from homogeneous coordinates
  p.px /= p.pw;
  p.py /= p.pw;
  p.pz /= p.pw;

  *x = p.px;
  *y = p.py;
  *z = p.pz;
}

void cylCoordinates(struct object3D *cyl, double a, double b, double *x, double *y, double *z)
{
 // Return in (x,y,z) the coordinates of a point on the plane given by the 2 parameters a,b.
 // 'a' in [0, 2*PI] corresponds to angle theta around the cylinder
 // 'b' in [0, 1] corresponds to height from the bottom
 
 /////////////////////////////////
 // TO DO: Complete this function.
 /////////////////////////////////

  // From Wikipedia
  // x = cos(theta)
  // y = sin(theta)

  point3D p;
  p.px = cos(a);
  p.py = sin(a);
  p.pz = b;
  p.pw = 1;
  matVecMult(cyl->T,&p);

  // Reset from homogeneous coordinates
  p.px /= p.pw;
  p.py /= p.pw;
  p.pz /= p.pw;

  *x = p.px;
  *y = p.py;
  *z = p.pz;
}

void planeSample(struct object3D *plane, double *x, double *y, double *z)
{
 // Returns the 3D coordinates (x,y,z) of a randomly sampled point on the plane
 // Sapling should be uniform, meaning there should be an equal change of gedtting
 // any spot on the plane

 /////////////////////////////////
 // TO DO: Complete this function.
 /////////////////////////////////   
  planeCoordinates(plane, drand48(), drand48(), x, y, z);
}

void sphereSample(struct object3D *sphere, double *x, double *y, double *z)
{
 // Returns the 3D coordinates (x,y,z) of a randomly sampled point on the sphere
 // Sampling should be uniform - note that this is tricky for a sphere, do some
 // research and document in your report what method is used to do this, along
 // with a reference to your source.
 
 /////////////////////////////////
 // TO DO: Complete this function.
 /////////////////////////////////   

  // 'a' in [0, 2*PI] corresponds to the spherical coordinate theta
  // 'b' in [-PI/2, PI/2] corresponds to the spherical coordinate phi
  sphereCoordinates(sphere,2*PI*drand48(),(PI*drand48())-(PI/2),x,y,z);
}

void cylSample(struct object3D *cyl, double *x, double *y, double *z)
{
 // Returns the 3D coordinates (x,y,z) of a randomly sampled point on the cylinder
 // Sampling should be uniform over the cylinder.

 /////////////////////////////////
 // TO DO: Complete this function.
 /////////////////////////////////   

 // 'a' in [0, 2*PI] corresponds to angle theta around the cylinder
 // 'b' in [0, 1] corresponds to height from the bottom
 cylCoordinates(cyl,2*PI*drand48(),drand48(),x,y,z);
}


/////////////////////////////////
// Texture mapping functions
/////////////////////////////////
void loadTexture(struct object3D *o, const char *filename, int type, struct textureNode **t_list)
{
 // Load a texture or normal map image from file and assign it to the
 // specified object. 
 // type:   1  ->  Texture map  (RGB, .ppm)
 //         2  ->  Normal map   (RGB, .ppm)
 //         3  ->  Alpha map    (grayscale, .pgm)
 // Stores loaded images in a linked list to avoid replication
 struct image *im;
 struct textureNode *p;
 
 if (o!=NULL)
 {
  // Check current linked list
  p=*(t_list);
  while (p!=NULL)
  {
   if (strcmp(&p->name[0],filename)==0)
   {
    // Found image already on the list
    if (type==1) o->texImg=p->im;
    else if (type==2) o->normalMap=p->im;
    else o->alphaMap=p->im;
    return;
   }
   p=p->next;
  }    

  // Load this texture image 
  if (type==1||type==2)
   im=readPPMimage(filename);
  else if (type==3)
   im=readPGMimage(filename);

  // Insert it into the texture list
  if (im!=NULL)
  {
   p=(struct textureNode *)calloc(1,sizeof(struct textureNode));
   strcpy(&p->name[0],filename);
   p->type=type;
   p->im=im;
   p->next=NULL; 
   // Insert into linked list
   if ((*(t_list))==NULL)
    *(t_list)=p;
   else
   {
    p->next=(*(t_list))->next;
    (*(t_list))->next=p;
   }
   // Assign to object
   if (type==1) o->texImg=im;
   else if (type==2) o->normalMap=im;
   else o->alphaMap=im;
  }
 
 }  // end if (o != NULL)
}

inline void getPixel(struct image *img, int offset, double *R, double *G, double *B)
{
  // Return the colour of a pixel
  if (img == NULL or img->rgbdata == NULL) {
    return;
  } else {
    //if (offset < img->sx*img->sy*3*sizeof(double)) return;
    *R = *((double *)img->rgbdata+offset);
    *G = *((double *)img->rgbdata+offset+1);
    *B = *((double *)img->rgbdata+offset+2);
  }
}

inline void getPixel_alpha(struct image *img, int offset, double *alpha)
{
  // Return the colour of a pixel
  if (img == NULL or img->rgbdata == NULL) {
    return;
  } else {
    *alpha = *((double *)img->rgbdata+offset);
  }
}

inline void bilinear_interpolation(struct image *img, double a, double b, double *R, double *G, double *B)
{
//http://www.cs.uu.nl/docs/vakken/gr/2011/Slides/06-texturing.pdf
  if (img == NULL) {
    return;
  } else {
    a = (a < 0) ? 0 : a;
    a = (a > 1) ? 1 : a;
    b = (b < 0) ? 0 : b;
    b = (b > 1) ? 1 : b;
    int i = floor(a*(img->sx-1));
    int j = floor(b*(img->sy-1));
    //int i2 = floor(a*img->sx+1);
    //i2 = (i2 <= img->sx) ? i2 : img->sx;
    //int j2 = floor(b*img->sy+1);
    //j2 = (j2 <= img->sy) ? j2 : img->sy;

    double du = a*(img->sx-1)-i;
    double dv = b*(img->sy-1)-j;
    int lim = img->sx*img->sy*3-1;

    double R00,R01,R10,R11;
    R00=R01=R10=R11=0;
    double G00,G01,G10,G11;
    G00=G01=G10=G11=0;
    double B00,B01,B10,B11;
    B00=B01=B10=B11=0;
    double c00,c01,c10,c11;

    // Bound cij by img size
    c00 = ((i*img->sy+j)*3 < lim) ? (i*img->sy+j)*3 : lim;
    c01 = ((i*img->sy+j+1)*3 < lim) ? (i*img->sy+j+1)*3 : lim;
    c10 = (((i+1)*img->sy+j)*3 < lim) ? ((i+1)*img->sy+j)*3 : lim;
    c11 = (((i+1)*img->sy+j+1)*3 < lim) ? ((i+1)*img->sy+j+1)*3 : lim;

    // Make sure we don't have negative numbers
    c00 = (c00 < 0) ? 0 : c00;
    c01 = (c01 < 0) ? 0 : c01;
    c10 = (c10 < 0) ? 0 : c10;
    c11 = (c11 < 0) ? 0 : c11;

    getPixel(img,c00,&R00,&G00,&B00);
    getPixel(img,c10,&R10,&G10,&B10);
    getPixel(img,c01,&R01,&G01,&B01);
    getPixel(img,c11,&R11,&G11,&B11);

    *R=(1-du)*(1-dv)*R00+du*(1-dv)*R10+(1-du)*dv*R01+du*dv*R11;
    *G=(1-du)*(1-dv)*G00+du*(1-dv)*G10+(1-du)*dv*G01+du*dv*G11;
    *B=(1-du)*(1-dv)*B00+du*(1-dv)*B10+(1-du)*dv*B01+du*dv*B11;
  }
}

void texMap(struct image *img, double a, double b, double *R, double *G, double *B)
{
 /*
  Function to determine the colour of a textured object at
  the normalized texture coordinates (a,b).

  a and b are texture coordinates in [0 1].
  img is a pointer to the image structure holding the texture for
   a given object.

  The colour is returned in R, G, B. Uses bi-linear interpolation
  to determine texture colour.
 */

 //////////////////////////////////////////////////
 // TO DO (Assignment 4 only):
 //
 //  Complete this function to return the colour
 // of the texture image at the specified texture
 // coordinates. Your code should use bi-linear
 // interpolation to obtain the texture colour.
 //////////////////////////////////////////////////

 bilinear_interpolation(img, a, b, R, G, B);
 //*(R)=1;  // Returns black - delete this and
 //*(G)=1;  // replace with your code to compute
 //*(B)=1;  // texture colour at (a,b)
 return;
}

void alphaMap(struct image *img, double a, double b, double *alpha)
{
 // Just like texture map but returns the alpha value at a,b,
 // notice that alpha maps are single layer grayscale images, hence
 // the separate function.

 //////////////////////////////////////////////////
 // TO DO (Assignment 4 only):
 //
 //  Complete this function to return the alpha
 // value from the image at the specified texture
 // coordinates. Your code should use bi-linear
 // interpolation to obtain the texture colour.
 //////////////////////////////////////////////////
 
  if (img == NULL) {
    return;
  } else {
    a = (a < 0) ? 0 : a;
    a = (a > 1) ? 1 : a;
    b = (b < 0) ? 0 : b;
    b = (b > 1) ? 1 : b;
    int i = floor(a*(img->sx-1));
    int j = floor(b*(img->sy-1));
    //int i2 = floor(a*img->sx+1);
    //i2 = (i2 <= img->sx) ? i2 : img->sx;
    //int j2 = floor(b*img->sy+1);
    //j2 = (j2 <= img->sy) ? j2 : img->sy;

    double du = a*(img->sx-1)-i;
    double dv = b*(img->sy-1)-j;
    int lim = img->sx*img->sy-1;

    double A00,A01,A10,A11;
    A00=A01=A10=A11=0;
    double c00,c01,c10,c11;

    // Bound cij by img size
    c00 = ((i*img->sy+j) < lim) ? (i*img->sy+j) : lim;
    c01 = ((i*img->sy+j+1) < lim) ? (i*img->sy+j+1) : lim;
    c10 = (((i+1)*img->sy+j) < lim) ? ((i+1)*img->sy+j) : lim;
    c11 = (((i+1)*img->sy+j+1) < lim) ? ((i+1)*img->sy+j+1) : lim;

    // Make sure we don't have negative numbers
    c00 = (c00 < 0) ? 0 : c00;
    c01 = (c01 < 0) ? 0 : c01;
    c10 = (c10 < 0) ? 0 : c10;
    c11 = (c11 < 0) ? 0 : c11;

    getPixel_alpha(img,c00,&A00);
    getPixel_alpha(img,c00,&A01);
    getPixel_alpha(img,c00,&A10);
    getPixel_alpha(img,c00,&A11);

    *(alpha)=(1-du)*(1-dv)*A00+du*(1-dv)*A10+(1-du)*dv*A01+du*dv*A11;
  }

 //*(alpha)=1;  // Returns 1 which means fully opaque. Replace
 return;  // with your code if implementing alpha maps.
}


/////////////////////////////
// Light sources
/////////////////////////////
void insertPLS(struct pointLS *l, struct pointLS **list)
{
 if (l==NULL) return;
 // Inserts a light source into the list of light sources
 if (*(list)==NULL)
 {
  *(list)=l;
  (*(list))->next=NULL;
 }
 else
 {
  l->next=(*(list))->next;
  (*(list))->next=l;
 }

}

void addAreaLight(double sx, double sy, double nx, double ny, double nz,\
                  double tx, double ty, double tz, int N,\
                  double r, double g, double b, struct object3D **o_list, struct pointLS **l_list)
{
 /*
   This function sets up and inserts a rectangular area light source
   with size (sx, sy)
   orientation given by the normal vector (nx, ny, nz)
   centered at (tx, ty, tz)
   consisting of (N) point light sources (uniformly sampled)
   and with colour/intensity (r,g,b) 
   
   Note that the light source must be visible as a uniformly colored rectangle which
   casts no shadows. If you require a lightsource to shade another, you must
   make it into a proper solid box with a back and sides of non-light-emitting
   materiale
 */

  /////////////////////////////////////////////////////
  // TO DO: (Assignment 4!)
  // Implement this function to enable area light sources
  /////////////////////////////////////////////////////

  // NOTE: The best way to implement area light sources is to random sample from the
  //       light source's object surface within rtShade(). This is a bit more tricky
  //       but reduces artifacts significantly. If you do that, then there is no need
  //       to insert a series of point lightsources in this function.

  struct object3D *area;
  struct point3D p;
  struct pointLS *l;
  double x,y,z;

  area = newPlane(1,1,1,1,r,g,b,1,1,1);
  area->isLightSource = 1;

  double theta = acos(nz / (sqrt(pow(nx, 2) + pow(ny, 2) + pow(nz, 2))));
  double phi = atan(ny / nx);

  Scale(area, sx, sy, sy);
  RotateZ(area, theta);
  RotateX(area, phi);
  Translate(area, tx, ty, tz);

  invert(&area->T[0][0], &area->Tinv[0][0]);
  insertObject(area, o_list);

  for (int i=0; i < N; i++) {
    // Randomly obtain points within the area
    planeSample(area, &x, &y, &z);
    p.px = x+(nx/10000);
    p.py = y+(ny/10000);
    p.pz = z+(nz/10000);
    p.pw = 1.0;
    l = newPLS(&p, r/N, g/N, b/N);  // Average out the intensity of each point
    insertPLS(l, l_list);
  }
}

///////////////////////////////////
// Geometric transformation section
///////////////////////////////////

void invert(double *T, double *Tinv)
{
 // Computes the inverse of transformation matrix T.
 // the result is returned in Tinv.

 double *U, *s, *V, *rv1;
 int singFlag, i;

 // Invert the affine transform
 U=NULL;
 s=NULL;
 V=NULL;
 rv1=NULL;
 singFlag=0;

 SVD(T,4,4,&U,&s,&V,&rv1);
 if (U==NULL||s==NULL||V==NULL)
 {
  fprintf(stderr,"Error: Matrix not invertible for this object, returning identity\n");
  memcpy(Tinv,eye4x4,16*sizeof(double));
  return;
 }

 // Check for singular matrices...
 for (i=0;i<4;i++) if (*(s+i)<1e-9) singFlag=1;
 if (singFlag)
 {
  fprintf(stderr,"Error: Transformation matrix is singular, returning identity\n");
  memcpy(Tinv,eye4x4,16*sizeof(double));
  return;
 }

 // Compute and store inverse matrix
 InvertMatrix(U,s,V,4,Tinv);

 free(U);
 free(s);
 free(V);
}

void RotateXMat(double T[4][4], double theta)
{
 // Multiply the current object transformation matrix T in object o
 // by a matrix that rotates the object theta *RADIANS* around the
 // X axis.

 double R[4][4];
 memset(&R[0][0],0,16*sizeof(double));

 R[0][0]=1.0;
 R[1][1]=cos(theta);
 R[1][2]=-sin(theta);
 R[2][1]=sin(theta);
 R[2][2]=cos(theta);
 R[3][3]=1.0;

 matMult(R,T);
}

void RotateX(struct object3D *o, double theta)
{
 // Multiply the current object transformation matrix T in object o
 // by a matrix that rotates the object theta *RADIANS* around the
 // X axis.

 double R[4][4];
 memset(&R[0][0],0,16*sizeof(double));

 R[0][0]=1.0;
 R[1][1]=cos(theta);
 R[1][2]=-sin(theta);
 R[2][1]=sin(theta);
 R[2][2]=cos(theta);
 R[3][3]=1.0;

 matMult(R,o->T);
}

void RotateYMat(double T[4][4], double theta)
{
 // Multiply the current object transformation matrix T in object o
 // by a matrix that rotates the object theta *RADIANS* around the
 // Y axis.

 double R[4][4];
 memset(&R[0][0],0,16*sizeof(double));

 R[0][0]=cos(theta);
 R[0][2]=sin(theta);
 R[1][1]=1.0;
 R[2][0]=-sin(theta);
 R[2][2]=cos(theta);
 R[3][3]=1.0;

 matMult(R,T);
}

void RotateY(struct object3D *o, double theta)
{
 // Multiply the current object transformation matrix T in object o
 // by a matrix that rotates the object theta *RADIANS* around the
 // Y axis.

 double R[4][4];
 memset(&R[0][0],0,16*sizeof(double));

 R[0][0]=cos(theta);
 R[0][2]=sin(theta);
 R[1][1]=1.0;
 R[2][0]=-sin(theta);
 R[2][2]=cos(theta);
 R[3][3]=1.0;

 matMult(R,o->T);
}

void RotateZMat(double T[4][4], double theta)
{
 // Multiply the current object transformation matrix T in object o
 // by a matrix that rotates the object theta *RADIANS* around the
 // Z axis.

 double R[4][4];
 memset(&R[0][0],0,16*sizeof(double));

 R[0][0]=cos(theta);
 R[0][1]=-sin(theta);
 R[1][0]=sin(theta);
 R[1][1]=cos(theta);
 R[2][2]=1.0;
 R[3][3]=1.0;

 matMult(R,T);
}

void RotateZ(struct object3D *o, double theta)
{
 // Multiply the current object transformation matrix T in object o
 // by a matrix that rotates the object theta *RADIANS* around the
 // Z axis.

 double R[4][4];
 memset(&R[0][0],0,16*sizeof(double));

 R[0][0]=cos(theta);
 R[0][1]=-sin(theta);
 R[1][0]=sin(theta);
 R[1][1]=cos(theta);
 R[2][2]=1.0;
 R[3][3]=1.0;

 matMult(R,o->T);
}

void TranslateMat(double T[4][4], double tx, double ty, double tz)
{
 // Multiply the current object transformation matrix T in object o
 // by a matrix that translates the object by the specified amounts.

 double tr[4][4];
 memset(&tr[0][0],0,16*sizeof(double));

 tr[0][0]=1.0;
 tr[1][1]=1.0;
 tr[2][2]=1.0;
 tr[0][3]=tx;
 tr[1][3]=ty;
 tr[2][3]=tz;
 tr[3][3]=1.0;

 matMult(tr,T);
}

void Translate(struct object3D *o, double tx, double ty, double tz)
{
 // Multiply the current object transformation matrix T in object o
 // by a matrix that translates the object by the specified amounts.

 double tr[4][4];
 memset(&tr[0][0],0,16*sizeof(double));

 tr[0][0]=1.0;
 tr[1][1]=1.0;
 tr[2][2]=1.0;
 tr[0][3]=tx;
 tr[1][3]=ty;
 tr[2][3]=tz;
 tr[3][3]=1.0;

 matMult(tr,o->T);
}

void ScaleMat(double T[4][4], double sx, double sy, double sz)
{
 // Multiply the current object transformation matrix T in object o
 // by a matrix that scales the object as indicated.

 double S[4][4];
 memset(&S[0][0],0,16*sizeof(double));

 S[0][0]=sx;
 S[1][1]=sy;
 S[2][2]=sz;
 S[3][3]=1.0;

 matMult(S,T);
}

void Scale(struct object3D *o, double sx, double sy, double sz)
{
 // Multiply the current object transformation matrix T in object o
 // by a matrix that scales the object as indicated.

 double S[4][4];
 memset(&S[0][0],0,16*sizeof(double));

 S[0][0]=sx;
 S[1][1]=sy;
 S[2][2]=sz;
 S[3][3]=1.0;

 matMult(S,o->T);
}

void printmatrix(double mat[4][4])
{
 fprintf(stderr,"Matrix contains:\n");
 fprintf(stderr,"%f %f %f %f\n",mat[0][0],mat[0][1],mat[0][2],mat[0][3]);
 fprintf(stderr,"%f %f %f %f\n",mat[1][0],mat[1][1],mat[1][2],mat[1][3]);
 fprintf(stderr,"%f %f %f %f\n",mat[2][0],mat[2][1],mat[2][2],mat[2][3]);
 fprintf(stderr,"%f %f %f %f\n",mat[3][0],mat[3][1],mat[3][2],mat[3][3]);
}

/////////////////////////////////////////
// Camera and view setup
/////////////////////////////////////////
struct view *setupView(struct point3D *e, struct point3D *g, struct point3D *up, double f, double wl, double wt, double wsize)
{
 /*
   This function sets up the camera axes and viewing direction as discussed in the
   lecture notes.
   e - Camera center
   g - Gaze direction
   up - Up vector
   fov - Fild of view in degrees
   f - focal length
 */
 struct view *c;
 struct point3D *u, *v;

 u=v=NULL;

 // Allocate space for the camera structure
 c=(struct view *)calloc(1,sizeof(struct view));
 if (c==NULL)
 {
  fprintf(stderr,"Out of memory setting up camera model!\n");
  return(NULL);
 }

 // Set up camera center and axes
 c->e.px=e->px;   // Copy camera center location, note we must make sure
 c->e.py=e->py;   // the camera center provided to this function has pw=1
 c->e.pz=e->pz;
 c->e.pw=1;

 // Set up w vector (camera's Z axis). w=-g/||g||
 c->w.px=-g->px;
 c->w.py=-g->py;
 c->w.pz=-g->pz;
 c->w.pw=1;
 normalize(&c->w);

 // Set up the horizontal direction, which must be perpenticular to w and up
 u=cross(&c->w, up);
 normalize(u);
 c->u.px=u->px;
 c->u.py=u->py;
 c->u.pz=u->pz;
 c->u.pw=1;

 // Set up the remaining direction, v=(u x w)  - Mind the signs
 v=cross(&c->u, &c->w);
 normalize(v);
 c->v.px=v->px;
 c->v.py=v->py;
 c->v.pz=v->pz;
 c->v.pw=1;

 // Copy focal length and window size parameters
 c->f=f;
 c->wl=wl;
 c->wt=wt;
 c->wsize=wsize;

 // Set up coordinate conversion matrices
 // Camera2World matrix (M_cw in the notes)
 // Mind the indexing convention [row][col]
 c->C2W[0][0]=c->u.px;
 c->C2W[1][0]=c->u.py;
 c->C2W[2][0]=c->u.pz;
 c->C2W[3][0]=0;

 c->C2W[0][1]=c->v.px;
 c->C2W[1][1]=c->v.py;
 c->C2W[2][1]=c->v.pz;
 c->C2W[3][1]=0;

 c->C2W[0][2]=c->w.px;
 c->C2W[1][2]=c->w.py;
 c->C2W[2][2]=c->w.pz;
 c->C2W[3][2]=0;

 c->C2W[0][3]=c->e.px;
 c->C2W[1][3]=c->e.py;
 c->C2W[2][3]=c->e.pz;
 c->C2W[3][3]=1;

 // World2Camera matrix (M_wc in the notes)
 // Mind the indexing convention [row][col]
 c->W2C[0][0]=c->u.px;
 c->W2C[1][0]=c->v.px;
 c->W2C[2][0]=c->w.px;
 c->W2C[3][0]=0;

 c->W2C[0][1]=c->u.py;
 c->W2C[1][1]=c->v.py;
 c->W2C[2][1]=c->w.py;
 c->W2C[3][1]=0;

 c->W2C[0][2]=c->u.pz;
 c->W2C[1][2]=c->v.pz;
 c->W2C[2][2]=c->w.pz;
 c->W2C[3][2]=0;

 c->W2C[0][3]=-dot(&c->u,&c->e);
 c->W2C[1][3]=-dot(&c->v,&c->e);
 c->W2C[2][3]=-dot(&c->w,&c->e);
 c->W2C[3][3]=1;

 free(u);
 free(v);
 return(c);
}

/////////////////////////////////////////
// Image I/O section
/////////////////////////////////////////
struct image *readPPMimage(const char *filename)
{
 // Reads an image from a .ppm file. A .ppm file is a very simple image representation
 // format with a text header followed by the binary RGB data at 24bits per pixel.
 // The header has the following form:
 //
 // P6
 // # One or more comment lines preceded by '#'
 // 340 200
 // 255
 //
 // The first line 'P6' is the .ppm format identifier, this is followed by one or more
 // lines with comments, typically used to inidicate which program generated the
 // .ppm file.
 // After the comments, a line with two integer values specifies the image resolution
 // as number of pixels in x and number of pixels in y.
 // The final line of the header stores the maximum value for pixels in the image,
 // usually 255.
 // After this last header line, binary data stores the RGB values for each pixel
 // in row-major order. Each pixel requires 3 bytes ordered R, G, and B.
 //
 // NOTE: Windows file handling is rather crotchetty. You may have to change the
 //       way this file is accessed if the images are being corrupted on read
 //       on Windows.
 //
 // readPPMdata converts the image colour information to floating point. This is so that
 // the texture mapping function doesn't have to do the conversion every time
 // it is asked to return the colour at a specific location.
 //

 FILE *f;
 struct image *im;
 char line[1024];
 int sizx,sizy;
 int i;
 unsigned char *tmp;
 double *fRGB;

 im=(struct image *)calloc(1,sizeof(struct image));
 if (im!=NULL)
 {
  im->rgbdata=NULL;
  f=fopen(filename,"rb+");
  if (f==NULL)
  {
   fprintf(stderr,"Unable to open file %s for reading, please check name and path\n",filename);
   free(im);
   return(NULL);
  }
  fgets(&line[0],1000,f);
  if (strcmp(&line[0],"P6\n")!=0)
  {
   fprintf(stderr,"Wrong file format, not a .ppm file or header end-of-line characters missing\n");
   free(im);
   fclose(f);
   return(NULL);
  }
  fprintf(stderr,"%s\n",line);
  // Skip over comments
  fgets(&line[0],511,f);
  while (line[0]=='#')
  {
   fprintf(stderr,"%s",line);
   fgets(&line[0],511,f);
  }
  sscanf(&line[0],"%d %d\n",&sizx,&sizy);           // Read file size
  fprintf(stderr,"nx=%d, ny=%d\n\n",sizx,sizy);
  im->sx=sizx;
  im->sy=sizy;

  fgets(&line[0],9,f);                    // Read the remaining header line
  fprintf(stderr,"%s\n",line);
  tmp=(unsigned char *)calloc(sizx*sizy*3,sizeof(unsigned char));
  fRGB=(double *)calloc(sizx*sizy*3,sizeof(double));
  if (tmp==NULL||fRGB==NULL)
  {
   fprintf(stderr,"Out of memory allocating space for image\n");
   free(im);
   fclose(f);
   return(NULL);
  }

  fread(tmp,sizx*sizy*3*sizeof(unsigned char),1,f);
  fclose(f);

  // Conversion to floating point
  for (i=0; i<sizx*sizy*3; i++) *(fRGB+i)=((double)*(tmp+i))/255.0;
  free(tmp);
  im->rgbdata=(void *)fRGB;

  return(im);
 }

 fprintf(stderr,"Unable to allocate memory for image structure\n");
 return(NULL);
}

struct image *readPGMimage(const char *filename)
{
 // Just like readPPMimage() except it is used to load grayscale alpha maps. In
 // alpha maps, a value of 255 corresponds to alpha=1 (fully opaque) and 0 
 // correspondst to alpha=0 (fully transparent).
 // A .pgm header of the following form is expected:
 //
 // P5
 // # One or more comment lines preceded by '#'
 // 340 200
 // 255
 //
 // readPGMdata converts the image grayscale data to double floating point in [0,1]. 

 FILE *f;
 struct image *im;
 char line[1024];
 int sizx,sizy;
 int i;
 unsigned char *tmp;
 double *fRGB;

 im=(struct image *)calloc(1,sizeof(struct image));
 if (im!=NULL)
 {
  im->rgbdata=NULL;
  f=fopen(filename,"rb+");
  if (f==NULL)
  {
   fprintf(stderr,"Unable to open file %s for reading, please check name and path\n",filename);
   free(im);
   return(NULL);
  }
  fgets(&line[0],1000,f);
  if (strcmp(&line[0],"P5\n")!=0)
  {
   fprintf(stderr,"Wrong file format, not a .pgm file or header end-of-line characters missing\n");
   free(im);
   fclose(f);
   return(NULL);
  }
  // Skip over comments
  fgets(&line[0],511,f);
  while (line[0]=='#')
   fgets(&line[0],511,f);
  sscanf(&line[0],"%d %d\n",&sizx,&sizy);           // Read file size
  im->sx=sizx;
  im->sy=sizy;

  fgets(&line[0],9,f);                    // Read the remaining header line
  tmp=(unsigned char *)calloc(sizx*sizy,sizeof(unsigned char));
  fRGB=(double *)calloc(sizx*sizy,sizeof(double));
  if (tmp==NULL||fRGB==NULL)
  {
   fprintf(stderr,"Out of memory allocating space for image\n");
   free(im);
   fclose(f);
   return(NULL);
  }

  fread(tmp,sizx*sizy*sizeof(unsigned char),1,f);
  fclose(f);

  // Conversion to double floating point
  for (i=0; i<sizx*sizy; i++) *(fRGB+i)=((double)*(tmp+i))/255.0;
  free(tmp);
  im->rgbdata=(void *)fRGB;

  return(im);
 }

 fprintf(stderr,"Unable to allocate memory for image structure\n");
 return(NULL);
}

struct image *newImage(int size_x, int size_y)
{
 // Allocates and returns a new image with all zeros. Assumes 24 bit per pixel,
 // unsigned char array.
 struct image *im;

 im=(struct image *)calloc(1,sizeof(struct image));
 if (im!=NULL)
 {
  im->rgbdata=NULL;
  im->sx=size_x;
  im->sy=size_y;
  im->rgbdata=(void *)calloc(size_x*size_y*3,sizeof(unsigned char));
  if (im->rgbdata!=NULL) return(im);
 }
 fprintf(stderr,"Unable to allocate memory for new image\n");
 return(NULL);
}

void imageOutput(struct image *im, const char *filename)
{
 // Writes out a .ppm file from the image data contained in 'im'.
 // Note that Windows typically doesn't know how to open .ppm
 // images. Use Gimp or any other seious image processing
 // software to display .ppm images.
 // Also, note that because of Windows file format management,
 // you may have to modify this file to get image output on
 // Windows machines to work properly.
 //
 // Assumes a 24 bit per pixel image stored as unsigned chars
 //

 FILE *f;

 if (im!=NULL)
  if (im->rgbdata!=NULL)
  {
   f=fopen(filename,"wb+");
   if (f==NULL)
   {
    fprintf(stderr,"Unable to open file %s for output! No image written\n",filename);
    return;
   }
   fprintf(f,"P6\n");
   fprintf(f,"# Output from RayTracer.c\n");
   fprintf(f,"%d %d\n",im->sx,im->sy);
   fprintf(f,"255\n");
   fwrite((unsigned char *)im->rgbdata,im->sx*im->sy*3*sizeof(unsigned char),1,f);
   fclose(f);
   return;
  }
 fprintf(stderr,"imageOutput(): Specified image is empty. Nothing output\n");
}

void deleteImage(struct image *im)
{
 // De-allocates memory reserved for the image stored in 'im'
 if (im!=NULL)
 {
  if (im->rgbdata!=NULL) free(im->rgbdata);
  free(im);
 }
}

void cleanup(struct object3D *o_list, struct pointLS *l_list, struct textureNode *t_list)
{
 // De-allocates memory reserved for the object list and the point light source
 // list. Note that *YOU* must de-allocate any memory reserved for images
 // rendered by the raytracer.
 struct object3D *p, *q;
 struct pointLS *r, *s;
 struct textureNode *t, *u;

 p=o_list;    // De-allocate all memory from objects in the list
 while(p!=NULL)
 {
  q=p->next;
  if (p->photonMap!=NULL) // If object is photon mapped, free photon map memory
  {
   if (p->photonMap->rgbdata!=NULL) free(p->photonMap->rgbdata);
   free(p->photonMap);
  }
  free(p);
  p=q;
 }

 r=l_list;    // Delete light source list
 while(r!=NULL)
 {
  s=r->next;
  free(r);
  r=s;
 }

 t=t_list;    // Delete texture Images
 while(t!=NULL)
 {
  u=t->next;
  if (t->im->rgbdata!=NULL) free(t->im->rgbdata);
  free(t->im);
  free(t);
  t=u;
 }
}

