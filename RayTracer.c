/*
  CSC D18 - RayTracer code.

  Written Dec. 9 2010 - Jan 20, 2011 by F. J. Estrada
  Freely distributable for adacemic purposes only.

  Uses Tom F. El-Maraghi's code for computing inverse
  matrices. You will need to compile together with
  svdDynamic.c

  You need to understand the code provided in
  this file, the corresponding header file, and the
  utils.c and utils.h files. Do not worry about
  svdDynamic.c, we need it only to compute
  inverse matrices.

  You only need to modify or add code in sections
  clearly marked "TO DO"

  Last updated: Aug. 12, 2014   - F.J.E.
*/

#include "utils.h"	// <-- This includes RayTracer.h
#include <time.h>


// A couple of global structures and data: An object list, a light list, and the
// maximum recursion depth
struct object3D *object_list;
struct pointLS *light_list;
struct textureNode *texture_list;
int MAX_DEPTH;
int AA_level = 4;  // Level of antialiasing; default 4xAA (i.e. AA_level=16 => 16x AA)
struct image *im; // Will hold the raytraced image
int antialiasing;  // Flag to determine whether antialiaing is enabled or disabled
int sx;    // Size of the raytraced image
int NUM_THREAD = 16;
int NUM_PIX_PER_THREAD;
int FOCAL_LENGTH = 30;  // For depth of field
int dof_samples = 16;

void buildScene(void)
{
 // Sets up all objects in the scene. This involves creating each object,
 // defining the transformations needed to shape and position it as
 // desired, specifying the reflectance properties (albedos and colours)
 // and setting up textures where needed.
 // Light sources must be defined, positioned, and their colour defined.
 // All objects must be inserted in the object_list. All light sources
 // must be inserted in the light_list.
 //
 // To create hierarchical objects:
 //    You must keep track of transformations carried out by parent objects
 //    as you move through the hierarchy. Declare and manipulate your own
 //    transformation matrices (use the provided functions in utils.c to
 //    compound transformations on these matrices). When declaring a new
 //    object within the hierarchy
 //    - Initialize the object
 //    - Apply any object-level transforms to shape/rotate/resize/move
 //      the object using regular object transformation functions
 //    - Apply the transformations passed on from the parent object
 //      by pre-multiplying the matrix containing the parent's transforms
 //      with the object's own transformation matrix.
 //    - Compute and store the object's inverse transform as usual.
 //
 // NOTE: After setting up the transformations for each object, don't
 //       forget to set up the inverse transform matrix!

 struct object3D *o;
 struct pointLS *l;
 struct point3D p;

 // Simple scene for Assignment 3:
 // Insert a couple of objects. A plane and two spheres
 // with some transformations.

 // Note the parameters: ra, rd, rs, rg, R, G, B, alpha, r_index, and shinyness)

/*
 int NUM_EARTH = 30;
 for (int i=0; i < NUM_EARTH; i++) {
   o=newSphere(.25,.75,.6,.2,.2,.95,.55,1,1,6);
   loadTexture(o,"./Texture/globe.ppm",1,&texture_list);
   RotateY(o,PI/3);
   Translate(o,-3+2.5*i,0,-5+2.5*i);
   Scale(o,2,2,2);
   invert(&o->T[0][0],&o->Tinv[0][0]);
   insertObject(o,&object_list);
 }*/
for (int i=0; i<30; i++) {
   struct object3D *k=newSphere(.25,.75,.6,.2,.2,.95,.55,1,1,6);
   loadTexture(k,"./Texture/globe.ppm",1,&texture_list);
   Translate(k,0,8,0);
   RotateX(k,i*PI/6);
   Translate(k,i*0.7,0,0);
   invert(&k->T[0][0],&k->Tinv[0][0]);
   insertObject(k,&object_list);
}
for (int i=1; i<30; i++) {
   struct object3D *k=newSphere(.25,.75,.6,.2,.2,.95,.55,1,1,6);
   loadTexture(k,"./Texture/globe.ppm",1,&texture_list);
   Translate(k,0,8,0);
   RotateX(k,-i*PI/6);
   Translate(k,-i*0.7,0,0);
   invert(&k->T[0][0],&k->Tinv[0][0]);
   insertObject(k,&object_list);
}

 o=newSphere(.25,.75,.6,.2,.85,.85,.75,1,1,8);
 loadTexture(o,"./Texture/moon.ppm",2,&texture_list);
 //loadTexture(o,"./Texture/alpha.pgm",3,&texture_list);
 Scale(o,5,5,5);
 Translate(o,6,0,0);
 invert(&o->T[0][0],&o->Tinv[0][0]);
 insertObject(o,&object_list);

 o=newSphere(.25,.75,.6,.2,.85,.85,.75,1,1,8);
 loadTexture(o,"./Texture/moon.ppm",2,&texture_list);
 //loadTexture(o,"./Texture/alpha.pgm",3,&texture_list);
 Scale(o,5,5,5);
 Translate(o,-6,0,0);
 invert(&o->T[0][0],&o->Tinv[0][0]);
 insertObject(o,&object_list);

 o=newSphere(0,0,1,.5,.9,.9,.9,.4,1.33,50);   // Initialize a sphere
 Scale(o,2,2,2);     
 Translate(o,0,0,-12); 
 invert(&o->T[0][0],&o->Tinv[0][0]);      // Compute the inverse transform * DON'T FORGET TO DO THIS! *
 insertObject(o,&object_list);

 o=newPlane(.05,.75,.05,.05,1,1,1,1,1,2);
 Scale(o,9000,9000,9000);
 //RotateZ(o,PI/4);
 RotateX(o,PI/2);
 Translate(o,0,-10,0);
 invert(&o->T[0][0],&o->Tinv[0][0]);
 insertObject(o,&object_list);

 // Insert a single point light source. We set up its position as a point structure, and specify its
 // colour in terms of RGB (in [0,1]).
 /*p.px=-2;
 p.py=10;
 p.pz=-5;
 p.pw=1;
 l=newPLS(&p,.95,.95,.95);
 insertPLS(l,&light_list);*/

 addAreaLight(10,10,0,-1.0,1,0,40,-40,300,.95,.95,.95,&object_list,&light_list);


 // End of simple scene for Assignment 3
 // Keep in mind that you can define new types of objects such as cylinders and parametric surfaces,
 // or, you can create code to handle arbitrary triangles and then define objects as surface meshes.
 //
 // Remember: A lot of the quality of your scene will depend on how much care you have put into defining
 //           the relflectance properties of your objects, and the number and type of light sources
 //           in the scene.
 
 ///////////////////////////////////////////////////////////////////////////////////////////////////////////
 // TO DO: For Assignment 4 you *MUST* define your own cool scene.
 //	   We will be looking for the quality of your scene setup, the use of hierarchical or composite
 //	   objects that are more interesting than the simple primitives from A3, the use of textures
 //        and other maps, illumination and illumination effects such as soft shadows, reflections and
 //        transparency, and the overall visual quality of your result. Put some work into thinking
 //        about these elements when designing your scene.
 ///////////////////////////////////////////////////////////////////////////////////////////////////////////
 
}

void rtShade(struct object3D *obj, struct point3D *p, struct point3D *n, struct ray3D *ray, int depth, double a, double b, struct colourRGB *col,\
  struct object3D *o_list, struct pointLS *l_list)
{
 // This function implements the shading model as described in lecture. It takes
 // - A pointer to the first object intersected by the ray (to get the colour properties)
 // - The coordinates of the intersection point (in world coordinates)
 // - The normal at the point
 // - The ray (needed to determine the reflection direction to use for the global component, as well as for
 //   the Phong specular component)
 // - The current racursion depth
 // - The (a,b) texture coordinates (meaningless unless texture is enabled)
 //
 // Returns:
 // - The colour for this ray (using the col pointer)
 //


 struct colourRGB tmp_col;	// Accumulator for colour components
 struct colourRGB refl_col;  // Accumulator for reflection colour components
 struct colourRGB refr_col;  // Accumulator for refraction colour components
 double R,G,B;			// Colour for the object in R G and B
 double alpha;

 // This will hold the colour as we process all the components of
 // the Phong illumination model
 tmp_col.R=0;
 tmp_col.G=0;
 tmp_col.B=0;

 refl_col.R=0;
 refl_col.G=0;
 refl_col.B=0;

 refr_col.R = 0;
 refr_col.G = 0;
 refr_col.B = 0;

 if (obj->texImg==NULL)		// Not textured, use object colour
 {
  R=obj->col.R;
  G=obj->col.G;
  B=obj->col.B;
 }/*
 else if (obj->normalMap)
 {
  obj->textureMap(obj->normalMap,a,b,&R,&G,&B);
 }*/
 else
 {
  // Get object colour from the texture given the texture coordinates (a,b), and the texturing function
  // for the object. Note that we will use textures also for Photon Mapping.
  obj->textureMap(obj->texImg,a,b,&R,&G,&B);
 }


 //////////////////////////////////////////////////////////////
 // TO DO: Implement this function. Refer to the notes for
 // details about the shading model.
 //////////////////////////////////////////////////////////////
 /*
 col->R = (n->px + 1) / 2;
 col->G = (n->py + 1) / 2;
 col->B = (n->pz + 1) / 2;
 */

 pointLS *light = l_list;
 ray3D light_ray;  // ray from intersection point to light source
 ray3D reflection_ray;  // ray reflected from object to perfect mirror direction
 ray3D refraction_ray;  // ray refracted by object
 point3D *light_d;  // direction vector from intersection point to light source
 double lambda;
 object3D *temp_obj;
 point3D temp_p, temp_n;
 double temp_a, temp_b;
 point3D s, de, m, ms;

 while (light != NULL) {
  /* Local Component */
  // construct ray from intersection point to light source
  light_ray.p0 = *p;
  light_ray.d = light->p0;
  subVectors(p,&light_ray.d);
  //normalize(&light_ray.d);

  // Ambient term
  tmp_col.R += (obj->alb.ra * light->col.R) * R;
  tmp_col.G += (obj->alb.ra * light->col.G) * G;
  tmp_col.B += (obj->alb.ra * light->col.B) * B;

  // Bounce the ray 
  findFirstHit(&light_ray,&lambda,obj,&temp_obj,&temp_p,&temp_n,&temp_a,&temp_b,o_list,l_list);
  if (lambda <= 0 || lambda >= 1) {
    // s: vector from intersection point to light source
    s.px = light->p0.px - p->px;
    s.py = light->p0.py - p->py;
    s.pz = light->p0.pz - p->pz;
    s.pw = 1;
    subVectors(p,&s);
    //normalize(s);

    // de: emittant direction vector
    de.px = -ray->d.px;
    de.py = -ray->d.py;
    de.pz = -ray->d.pz;
    de.pw = 1;
    //normalize(de);

    // m: 2(s.n)n-s, perfect mirror direction vector
    m.px = 2 * dot(&s,n) * n->px;
    m.py = 2 * dot(&s,n) * n->py;
    m.pz = 2 * dot(&s,n) * n->pz;
    m.pw = 1;
    subVectors(&s,&m);
    normalize(&m);
    normalize(n);
    normalize(&s);

    // Diffuse term
    tmp_col.R += (obj->alb.rd * light->col.R * max(0,dot(&s,n))) * R;
    tmp_col.G += (obj->alb.rd * light->col.G * max(0,dot(&s,n))) * G;
    tmp_col.B += (obj->alb.rd * light->col.B * max(0,dot(&s,n))) * B;

    // Specular terms
    tmp_col.R += (obj->alb.rs * light->col.R * pow(max(0,dot(&m,&de)), obj->shinyness));
    tmp_col.G += (obj->alb.rs * light->col.G * pow(max(0,dot(&m,&de)), obj->shinyness));
    tmp_col.B += (obj->alb.rs * light->col.B * pow(max(0,dot(&m,&de)), obj->shinyness));
  }

  /* Global Component */
  if (depth < MAX_DEPTH) {
    // Global specular reflection
    if (obj->alb.rs > 0) {
      // mirror direction vector
      ms.px = ray->d.px - (2 * dot(n,&ray->d) * n->px);
      ms.py = ray->d.py - (2 * dot(n,&ray->d) * n->py);
      ms.pz = ray->d.pz - (2 * dot(n,&ray->d) * n->pz);
      ms.pw = 1;
      normalize(&ms);

      // construct reflection ray
      reflection_ray.p0 = *p;
      reflection_ray.d = ms;

      // raytrace recursion on reflection ray
      rayTrace(&reflection_ray,depth++,&refl_col,obj,o_list,l_list);

      // Update specular reflection term
      tmp_col.R += (obj->alb.rg * refl_col.R * R);
      tmp_col.G += (obj->alb.rg * refl_col.G * G);
      tmp_col.B += (obj->alb.rg * refl_col.B * B);
    }

    if (obj->alphaMap) {
      alphaMap(obj->alphaMap,a,b,&alpha);
    }

    // Refraction
    if (obj->alpha < 1 || (obj->alphaMap && alpha < 1)) {
      struct point3D b;

      b.px = -ray->d.px;
      b.py = -ray->d.py;
      b.pz = -ray->d.pz;
      b.pw = 1;

      double r;

      if (dot(n,&b) > 0) {
        r = 1 / obj->r_index;
      } else {
        r = obj->r_index;
        n->px = -n->px;
        n->py = -n->py;
        n->pz = -n->pz;
      }

      double c1 = dot(n,&b);
      double c2 = r*r * (1-pow(c1,2));

      if (c2 <= 1) {
        struct point3D refr;
        double c3 = sqrt(1-c2);
        refr.px = r * b.px - (r * c1 + c3) * n->px;
        refr.py = r * b.py - (r * c1 + c3) * n->py;
        refr.pz = r * b.pz - (r * c1 + c3) * n->pz;
        refr.pw = 1;
        
        normalize(&refr);
        refraction_ray.d = refr;
        refraction_ray.p0 = *p;

        rayTrace(&refraction_ray,depth++,&refr_col,obj,o_list,l_list);
      }

      /*tmp_col.R = (tmp_col.R * (1 - obj->alpha)) + (obj->alpha * R * refr_col.R);
      tmp_col.G = (tmp_col.G * (1 - obj->alpha)) + (obj->alpha * G * refr_col.G);
      tmp_col.B = (tmp_col.B * (1 - obj->alpha)) + (obj->alpha * B * refr_col.B);*/
      if (obj->alphaMap) {
        tmp_col.R = (tmp_col.R * (alpha)) + ((1-alpha) * R * refr_col.R);
        tmp_col.G = (tmp_col.G * (alpha)) + ((1-alpha) * G * refr_col.G);
        tmp_col.B = (tmp_col.B * (alpha)) + ((1-alpha) * B * refr_col.B);
      } else {
        tmp_col.R = (tmp_col.R * (obj->alpha)) + ((1-obj->alpha) * R * refr_col.R);
        tmp_col.G = (tmp_col.G * (obj->alpha)) + ((1-obj->alpha) * G * refr_col.G);
        tmp_col.B = (tmp_col.B * (obj->alpha)) + ((1-obj->alpha) * B * refr_col.B);
      }
    }
  }
  light = light->next;
 }
/*
 // Alpha Mapping
 if (obj->alphaMap) {
  alphaMap(obj->alphaMap,a,b,&alpha);
 } else {*/
  alpha = 1.0;
 //}
 // Update col
 col->R = tmp_col.R * alpha;
 col->G = tmp_col.G * alpha;
 col->B = tmp_col.B * alpha;

 // Bounds on Colour
 if (col->R > 1) col->R = 1;
 if (col->G > 1) col->G = 1;
 if (col->B > 1) col->B = 1;

 // Be sure to update 'col' with the final colour computed here!
 return;
}

void findFirstHit(struct ray3D *ray, double *lambda, struct object3D *Os, struct object3D **obj, struct point3D *p, \
  struct point3D *n, double *a, double *b, struct object3D *o_list, struct pointLS *l_list)
{
 // Find the closest intersection between the ray and any objects in the scene.
 // Inputs:
 //   *ray    -  A pointer to the ray being traced
 //   *Os     -  'Object source' is a pointer toward the object from which the ray originates. It is used for reflected or refracted rays
 //              so that you can check for and ignore self-intersections as needed. It is NULL for rays originating at the center of
 //              projection
 // Outputs:
 //   *lambda -  A pointer toward a double variable 'lambda' used to return the lambda at the intersection point
 //   **obj   -  A pointer toward an (object3D *) variable so you can return a pointer to the object that has the closest intersection with
 //              this ray (this is required so you can do the shading)
 //   *p      -  A pointer to a 3D point structure so you can store the coordinates of the intersection point
 //   *n      -  A pointer to a 3D point structure so you can return the normal at the intersection point
 //   *a, *b  -  Pointers toward double variables so you can return the texture coordinates a,b at the intersection point

 /////////////////////////////////////////////////////////////
 // TO DO: Implement this function. See the notes for
 // reference of what to do in here
 /////////////////////////////////////////////////////////////

  double hit_lambda = -1;
  double hit_a, hit_b;
  point3D hit_p, hit_n;

  // Intersect the ray to all objects
  object3D *currentObject = o_list;
  while (currentObject != NULL) {
    if (currentObject != Os) {
      // Find closest legitimate intersection of ray and object
      currentObject->intersect(currentObject,ray,lambda,&hit_p,&hit_n,&hit_a,&hit_b);
      if ((hit_lambda < 0 || *lambda < hit_lambda) && (*lambda > 0)) {
        // Find smallest lambda = closest intersection
        hit_lambda = *lambda;
        *obj = currentObject;
        *p = hit_p;
        *n = hit_n;
        *a = hit_a;
        *b = hit_b;
      }
    }
    currentObject = currentObject->next;
  }
  *lambda = hit_lambda;
}

void rayTrace(struct ray3D *ray, int depth, struct colourRGB *col, struct object3D *Os,\
  struct object3D *o_list, struct pointLS *l_list)
{
 // Trace one ray through the scene.
 //
 // Parameters:
 //   *ray   -  A pointer to the ray being traced
 //   depth  -  Current recursion depth for recursive raytracing
 //   *col   - Pointer to an RGB colour structure so you can return the object colour
 //            at the intersection point of this ray with the closest scene object.
 //   *Os    - 'Object source' is a pointer to the object from which the ray 
 //            originates so you can discard self-intersections due to numerical
 //            errors. NULL for rays originating from the center of projection. 
 
 double lambda;		// Lambda at intersection
 double a,b;		// Texture coordinates
 struct object3D *obj;	// Pointer to object at intersection
 struct point3D p;	// Intersection point
 struct point3D n;	// Normal at intersection
 struct colourRGB I;	// Colour returned by shading function

 if (depth>MAX_DEPTH)	// Max recursion depth reached. Return invalid colour.
 {
  col->R=-1;
  col->G=-1;
  col->B=-1;
  return;
 }

 ///////////////////////////////////////////////////////
 // TO DO: Complete this function. Refer to the notes
 // if you are unsure what to do here.
 ///////////////////////////////////////////////////////

 // Shine the first ray and test for first intersection point
 findFirstHit(ray,&lambda,Os,&obj,&p,&n,&a,&b,o_list,l_list);
 if (lambda > 0) {
  // If intersection point exists then compute shading model
  rtShade(obj,&p,&n,ray,depth,a,b,col,o_list,l_list);
 } else {
  // no hit, set to background color
  col->R = 0;
  col->G = 0;
  col->B = 0;
 }

}

void* trace_thread(void *ptr)
{
  /* Multi-threading support - this function contains the tasks for each thread */

  struct thread_data *data;
  // Retrieve data passed to the this thread
  data = (struct thread_data *) ptr;

  int i,j;
  int offset;
  struct point3D pc,d;   // Point structures to keep the coordinates of a pixel and
        // the direction or a ray
  //struct ray3D *ray;   // Structure to keep the ray from e to a pixel
  struct colourRGB col;    // Return colour for raytraced pixels
  struct colourRGB AA_col;   // Colour for antialiasing
  struct colourRGB dof_col;
  struct point3D focal_point, jittered_point, new_d;

  //fprintf(stderr," %d ", data->j);
  struct ray3D *ray = (struct ray3D *)malloc(sizeof(struct ray3D));

  for (int x=0;x<NUM_PIX_PER_THREAD;x++) {
    // Pixel (i, j)
    i = (x + NUM_PIX_PER_THREAD * data->thread_id) / sx;
    j = (x + NUM_PIX_PER_THREAD * data->thread_id) % sx;

    if ((i*sx+j)*3 > sx*sx*3) break; // just in case this happens
    if (j == 0) fprintf(stderr, "%d/%d, ", i,sx);
  
    if (!antialiasing) {
      // pc_ij = (wt+i*du, wt+j*dv, f)
      pc.px = data->cam->wl + (i * data->du);
      pc.py = data->cam->wt + (j * data->dv);
      pc.pz = data->cam->f;
      pc.pw = 1;

      // pw_ij = [u v w]pc_ij
      matVecMult(data->cam->C2W,&pc);

      // d_ij = pw_ij - ew
      d.px = pc.px;
      d.py = pc.py;
      d.pz = pc.pz;
      d.pw = 1;

      subVectors(&data->e,&d);
      normalize(&d);
/*
      // Depth of field
      focal_point.px = pc.px + FOCAL_LENGTH * d.px;
      focal_point.py = pc.py + FOCAL_LENGTH * d.py;
      focal_point.pz = pc.pz + FOCAL_LENGTH * d.pz;
      focal_point.pw = 1.0;

      for (int z=0; z < dof_samples; z++) {
        memcpy(&new_d,&focal_point,sizeof(struct point3D));

        jittered_point.px = (drand48()-0.5) / 50;
        jittered_point.py = (drand48()-0.5) / 50;
        jittered_point.pz = -20.0;
        jittered_point.pw = 1.0;

        memcpy(&ray->p0,&jittered_point,sizeof(struct point3D));
        subVectors(&jittered_point,&new_d);
        normalize(&new_d);
        memcpy(&ray->d,&new_d,sizeof(struct point3D));

        // r(\lambda) = pw_ij + d_ij * \lambda
        //fprintf(stderr, "r=[%f,%f,%f]+lambda*[%f,%f,%f]\n", pc.px,pc.py,pc.pz,d.px,d.py,d.pz);
*/ 
        memcpy(&ray->p0,&pc,sizeof(struct point3D));
        memcpy(&ray->d,&d,sizeof(struct point3D));
        rayTrace(ray,1,&col,NULL,data->object_list,data->light_list);
/*
        // Preset colour
        dof_col.R = data->background.R;
        dof_col.G = data->background.G;
        dof_col.B = data->background.B;

        rayTrace(ray,1,&dof_col,NULL,data->object_list,data->light_list);
        col.R += dof_col.R;
        col.G += dof_col.G;
        col.B += dof_col.B;
      }
      col.R /= dof_samples;
      col.G /= dof_samples;
      col.B /= dof_samples;
      */
      // R,G,B each pixel, sx pixels per row
      offset = (i * 3) + (j * sx * 3);
      *((unsigned char *)im->rgbdata + offset) = col.R * 255;
      *((unsigned char *)im->rgbdata + offset + 1) = col.G * 255;
      *((unsigned char *)im->rgbdata + offset + 2) = col.B * 255;
    } else {
      AA_col.R = 0;
      AA_col.G = 0;
      AA_col.B = 0;

      for (int x = 0; x < AA_level; x++) {
        // Divide pixel into smaller pixels and shine rays through them

        // pc_ij = (wt+i*du, wt+j*dv, f)
        //pc.px = cam->wl + (i * du);
        pc.px = data->cam->wl + ((i+drand48()-0.5) * data->du);
        //pc.py = cam->wt + (j * dv);
        pc.py = data->cam->wt + ((j+drand48()-0.5) * data->dv);
        pc.pz = data->cam->f;
        pc.pw = 1;

        // pw_ij = [u v w]pc_ij
        matVecMult(data->cam->C2W,&pc);

        // d_ij = pw_ij - ew
        d.px = pc.px;
        d.py = pc.py;
        d.pz = pc.pz;
        d.pw = 1;
        subVectors(&data->e,&d);
        normalize(&d);
/*
        // Depth of field
        focal_point.px = pc.px + FOCAL_LENGTH * d.px;
        focal_point.py = pc.py + FOCAL_LENGTH * d.py;
        focal_point.pz = pc.pz + FOCAL_LENGTH * d.pz;
        focal_point.pw = 1.0;

        for (int z=0; z < dof_samples; z++) {
          memcpy(&new_d,&focal_point,sizeof(struct point3D));

          jittered_point.px = (drand48()-0.5) / 50;
          jittered_point.py = (drand48()-0.5) / 50;
          jittered_point.pz = -20.0;
          jittered_point.pw = 1.0;

          memcpy(&ray->p0,&jittered_point,sizeof(struct point3D));
          subVectors(&jittered_point,&new_d);
          normalize(&new_d);
          memcpy(&ray->d,&new_d,sizeof(struct point3D));

          // r(\lambda) = pw_ij + d_ij * \lambda
          //fprintf(stderr, "r=[%f,%f,%f]+lambda*[%f,%f,%f]\n", pc.px,pc.py,pc.pz,d.px,d.py,d.pz);
          memcpy(&ray->p0,&pc,sizeof(struct point3D));
          memcpy(&ray->d,&d,sizeof(struct point3D));

          // Preset colour
          dof_col.R = data->background.R;
          dof_col.G = data->background.G;
          dof_col.B = data->background.B;

          rayTrace(ray,1,&dof_col,NULL,data->object_list,data->light_list);
          col.R += dof_col.R;
          col.G += dof_col.G;
          col.B += dof_col.B;
        }
        col.R /= dof_samples;
        col.G /= dof_samples;
        col.B /= dof_samples;*/

        // r(\lambda) = pw_ij + d_ij * \lambda
        //fprintf(stderr, "r=[%f,%f,%f]+lambda*[%f,%f,%f]\n", pc.px,pc.py,pc.pz,d.px,d.py,d.pz);
        memcpy(&ray->p0,&pc,sizeof(struct point3D));
        memcpy(&ray->d,&d,sizeof(struct point3D));

        col.R = data->background.R;
        col.G = data->background.G;
        col.B = data->background.B;

        //rayTrace(struct ray3D *ray, int depth, struct colourRGB *col, struct object3D *Os)
        rayTrace(ray,1,&col,NULL,data->object_list,data->light_list);

        AA_col.R += col.R;
        AA_col.G += col.G;
        AA_col.B += col.B;
      }
      // R,G,B each pixel, sx pixels per row
      offset = (i * 3) + (j * sx * 3);
      // Get the average of the Colours of all the antialiasing pixels
      AA_col.R /= AA_level;
      AA_col.G /= AA_level;
      AA_col.B /= AA_level;
      *((unsigned char *)im->rgbdata + offset) = AA_col.R * 255;
      *((unsigned char *)im->rgbdata + offset + 1) = AA_col.G * 255;
      *((unsigned char *)im->rgbdata + offset + 2) = AA_col.B * 255;
    }
  //} // end for i
  }
  free(ray);
  pthread_exit(0);
}

int main(int argc, char *argv[])
{
 // Main function for the raytracer. Parses input parameters,
 // sets up the initial blank image, and calls the functions
 // that set up the scene and do the raytracing.
 
 struct view *cam;	// Camera and view for this scene
 char output_name[1024];	// Name of the output file for the raytraced .ppm image
 struct point3D e;		// Camera view parameters 'e', 'g', and 'up'
 struct point3D g;
 struct point3D up;
 double du, dv;			// Increase along u and v directions for pixel coordinates
 struct colourRGB background;   // Background colour
 int j;			// Counters for pixel coordinates
 unsigned char *rgbIm;

 if (argc<5)
 {
  fprintf(stderr,"RayTracer: Can not parse input parameters\n");
  fprintf(stderr,"USAGE: RayTracer size rec_depth antialias output_name\n");
  fprintf(stderr,"   size = Image size (both along x and y)\n");
  fprintf(stderr,"   rec_depth = Recursion depth\n");
  fprintf(stderr,"   antialias = A single digit, 0 disables antialiasing. Anything else enables antialiasing\n");
  fprintf(stderr,"   output_name = Name of the output file, e.g. MyRender.ppm\n");
  exit(0);
 }
 sx=atoi(argv[1]);
 MAX_DEPTH=atoi(argv[2]);
 if (atoi(argv[3])==0) antialiasing=0; else antialiasing=1;
 strcpy(&output_name[0],argv[4]);
 if (argc == 6) AA_level=atoi(argv[5]);

 fprintf(stderr,"Rendering image at %d x %d\n",sx,sx);
 fprintf(stderr,"Recursion depth = %d\n",MAX_DEPTH);
 if (!antialiasing) fprintf(stderr,"Antialising is off\n");
 else {
  fprintf(stderr,"Antialising is on\n");
  fprintf(stderr,"Level of antialising: %dx\n",AA_level);
 }
 fprintf(stderr,"Output file name: %s\n",output_name);

 object_list=NULL;
 light_list=NULL;
 texture_list=NULL;

 // Allocate memory for the new image
 im=newImage(sx, sx);
 if (!im)
 {
  fprintf(stderr,"Unable to allocate memory for raytraced image\n");
  exit(0);
 }
 else rgbIm=(unsigned char *)im->rgbdata;

 ///////////////////////////////////////////////////
 // TO DO: You will need to implement several of the
 //        functions below. For Assignment 3, you can use
 //        the simple scene already provided. But
 //        for Assignment 4 you need to create your own
 //        *interesting* scene.
 ///////////////////////////////////////////////////
 buildScene();		// Create a scene. This defines all the
			// objects in the world of the raytracer

 //////////////////////////////////////////
 // TO DO: For Assignment 3 you can use the setup
 //        already provided here. For Assignment 4
 //        you may want to move the camera
 //        and change the view parameters
 //        to suit your scene.
 //////////////////////////////////////////

 // Mind the homogeneous coordinate w of all vectors below. DO NOT
 // forget to set it to 1, or you'll get junk out of the
 // geometric transformations later on.

 // Camera center is at (0,0,-1)
 e.px=0;
 e.py=0;
 e.pz=-20;
 e.pw=1;

 // To define the gaze vector, we choose a point 'pc' in the scene that
 // the camera is looking at, and do the vector subtraction pc-e.
 // Here we set up the camera to be looking at the origin.
 g.px=0-e.px;
 g.py=0-e.py;
 g.pz=0-e.pz;
 g.pw=1;
 // In this case, the camera is looking along the world Z axis, so
 // vector w should end up being [0, 0, -1]

 // Define the 'up' vector to be the Y axis
 up.px=0;
 up.py=1;
 up.pz=0;
 up.pw=1;

 // Set up view with given the above vectors, a 4x4 window,
 // and a focal length of -1 (why? where is the image plane?)
 // Note that the top-left corner of the window is at (-2, 2)
 // in camera coordinates.
 cam=setupView(&e, &g, &up, -5, -5, 5, 10);

 if (cam==NULL)
 {
  fprintf(stderr,"Unable to set up the view and camera parameters. Our of memory!\n");
  cleanup(object_list,light_list, texture_list);
  deleteImage(im);
  exit(0);
 }

 // Set up background colour here
 background.R=0;
 background.G=0;
 background.B=0;

 // Do the raytracing
 //////////////////////////////////////////////////////
 // TO DO: You will need code here to do the raytracing
 //        for each pixel in the image. Refer to the
 //        lecture notes, in particular, to the
 //        raytracing pseudocode, for details on what
 //        to do here. Make sure you undersand the
 //        overall procedure of raytracing for a single
 //        pixel.
 //////////////////////////////////////////////////////
 du=cam->wsize/(sx-1);		// du and dv. In the notes in terms of wl and wr, wt and wb,
 dv=-cam->wsize/(sx-1);		// here we use wl, wt, and wsize. du=dv since the image is
				// and dv is negative since y increases downward in pixel
				// coordinates and upward in camera coordinates.

 fprintf(stderr,"View parameters:\n");
 fprintf(stderr,"Left=%f, Top=%f, Width=%f, f=%f\n",cam->wl,cam->wt,cam->wsize,cam->f);
 fprintf(stderr,"Camera to world conversion matrix (make sure it makes sense!):\n");
 printmatrix(cam->C2W);
 fprintf(stderr,"World to camera conversion matrix:\n");
 printmatrix(cam->W2C);
 fprintf(stderr,"\n");

 // For render time measuring
 struct timespec start, finish;
 double time_spent;
 clock_gettime(CLOCK_MONOTONIC, &start);
 //fprintf(stderr,"Rendering row: ");

 pthread_t threads[NUM_THREAD];
 struct thread_data data[NUM_THREAD];  // data to be passed to threads

 // Evenly divide the amount of pixels to render for each thread
 NUM_PIX_PER_THREAD = sx * sx / NUM_THREAD;

 for (int x=0; x < NUM_THREAD; x++) {
  //data = (struct thread_data *)malloc(sizeof(struct thread_data));
  data[x].thread_id = x;
  data[x].du = du;
  data[x].dv = dv;
  memcpy(&data[x].e,&e,sizeof(struct point3D));
  data[x].cam = (struct view *)malloc(sizeof(struct view));
  memcpy(data[x].cam,cam,sizeof(struct view));
  memcpy(&data[x].background,&background,sizeof(struct colourRGB));

  // These global structures become part of the thread data structure
  // such that threads don't have to wait in order to access the structures.
  data[x].object_list = (struct object3D *)malloc(sizeof(struct object3D));
  memcpy(data[x].object_list,object_list,sizeof(struct object3D));

  data[x].light_list = (struct pointLS *)malloc(sizeof(struct pointLS));
  memcpy(data[x].light_list,light_list,sizeof(struct pointLS));
  //memcpy(&data[x].rgbIm,&rgbIm,sizeof(unsigned char));

  // multi-threading, each of the NUM_THREAD threads trace NUM_PIX_PER_THREAD pixels
  pthread_create(&threads[x],NULL,&trace_thread, &data[x]);
  fprintf(stderr, "Thread created: %d\n", x);
 }

 for (j=0;j<NUM_THREAD;j++)
 {
  // wait until all threads are completed
  pthread_join(threads[j],NULL);
  fprintf(stderr, "Thread completed: %d\n", j);
 }
 for (j=0;j<NUM_THREAD;j++)
 {
  // Free allocated memory
  free(data[j].cam);
  free(data[j].object_list);
  free(data[j].light_list);
 }
 fprintf(stderr,"\nDone!\n");

 // Output rendered image
 imageOutput(im,output_name);

 clock_gettime(CLOCK_MONOTONIC,&finish);
 time_spent= (finish.tv_sec - start.tv_sec);
 fprintf(stderr, "Render time: %f seconds\n", time_spent);
 
 // Exit section. Clean up and return.
 cleanup(object_list,light_list,texture_list);		// Object, light, and texture lists
 deleteImage(im);					// Rendered image
 free(cam);						// camera view
 exit(0);
}

