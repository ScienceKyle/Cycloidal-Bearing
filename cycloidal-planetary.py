import numpy as np
import trimesh
from scipy.interpolate import interp1d
from shapely.geometry import Polygon
from trimesh.transformations import rotation_matrix, translation_matrix


bool_engine = "manifold"

def generate_planet(N=6, base_r=0.5, height=0.5, segments=360):
    r = base_r / (2 * N)
    th = np.linspace(0, np.pi/N, 360)
    
    # Generate one tooth shape (epitrochoid + hypotrochoid)
    epi = (base_r + r)*np.column_stack([np.cos(th), np.sin(th)]) - \
          r*np.column_stack([np.cos(th*(base_r + r)/r), np.sin(th*(base_r + r)/r)])
    hyp = (base_r - r)*np.column_stack([np.cos(th + np.pi/N), np.sin(th + np.pi/N)]) + \
          r*np.column_stack([np.cos((base_r - r)*(th + np.pi/N)/r), -np.sin((base_r - r)*(th + np.pi/N)/r)])
    tooth = np.concatenate([epi, hyp])

    # Rotate tooth around N times
    phases = np.linspace(0, 2*np.pi, N, endpoint=False)
    rot = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    points = np.concatenate([tooth @ rot(p).T for p in phases])

    # Resample along the perimeter
    dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    uniform_s = np.linspace(0, dist[-1], segments + 1)
    x = interp1d(dist, points[:, 0])(uniform_s)
    y = interp1d(dist, points[:, 1])(uniform_s)
    
    radius = np.sqrt(x**2+y**2)+.003
    theta = np.arctan2(x,y)
    
    x2 = radius*np.cos(theta)
    y2 = radius*np.sin(theta)
    cycloid = Polygon(np.column_stack((x, y)))
    cycloid2 = Polygon(np.column_stack((x2, y2)))
    
    mesh = trimesh.creation.extrude_polygon(polygon=cycloid,height=height,cap=True)
    mesh2 = trimesh.creation.extrude_polygon(polygon=cycloid2,height=height,cap=True)
    
  
    barrel_start = base_r-5*r
    barrel_max = base_r+4*r
    
    profile = np.array([
        [0.0,     0.0],
        [barrel_start,0.0],
        [barrel_max,  0.35 * height],
        [barrel_max,  0.65 * height],
        [barrel_start,      height],
        [0.0,     height]
    ])
    
    barrel_start2 = base_r-5*r+.003
    barrel_max2 = base_r+4*r+.003

    
    profile2 = np.array([
        [0.0,     0.0],
        [barrel_start2,0.0],
        [barrel_max2,  0.35 * height],
        [barrel_max2,  0.65 * height],
        [barrel_start2,      height],
        [0.0,     height]
    ])
    
    # Revolve the profile around the Y-axis
    barrel = trimesh.creation.revolve(profile, angle=np.pi * 2, segments=64,cap=True)
    barrel2 = trimesh.creation.revolve(profile2, angle=np.pi * 2, segments=64,cap=True)

    return repair(mesh.union(repair(barrel2),engine='manifold')) , repair(mesh.union(repair(barrel),engine='manifold'))

def repair(mesh=None):
    mesh.update_faces(mesh.unique_faces())
    mesh = mesh.process(validate=True)

    return mesh
 
def generate_plantery(id=2.5,od=4,height=.75):
  
  N_p = 8
  N_r = 44
  N_s = 28
  num_p = 12
  best_r = 1/3
  gap = .6*(od-id)/2
  orbit_r = (od+id)/4
  orbit_D = 2*np.pi*orbit_r
  thresh = .001
  flag = False
  possible_base_r = np.linspace(gap/2,gap/4,10000)
  
  for r in possible_base_r:
    for N_p in [12,11,10,9,8,7,6,5]:
      N_r_exact = N_p * (orbit_r + r) / r
      N_s_exact = N_p * (orbit_r - r) / r
      loop_N_r = round(N_r_exact)
      loop_N_s = round(N_s_exact)
      ring_error = abs(N_r_exact - loop_N_r)
      sun_error = abs(N_s_exact - loop_N_s)
       
      if ring_error > thresh and sun_error > thresh:
        continue
      factors = np.arange(2, int(np.floor(orbit_D/(2*r+2*r/N_p)))+1)
      
        #factors = np.arange(3, 16)
      valid_factors = factors[(loop_N_r+loop_N_s) % factors == 0]
      packing = 2*r*np.max(valid_factors)/orbit_D
      
      if len(valid_factors) == 0 or not np.max(valid_factors) % 2 == 0  or 6 not in valid_factors or packing<.75:
          continue
      
      best_r = r
      N_r = loop_N_r
      N_s = loop_N_s
      num_p = np.max(valid_factors)
      flag = True
      break
    if flag:
      break
    
  if not flag:
    print("No solutions")
    return
    
  print(f'N_p={N_p},N_r={N_r},N_s={N_s},num={num_p},r={best_r}')
  
  ring_rat = N_r/N_p
  sun_rat = N_s/N_p
  
  planet_out, planet = generate_planet(N_p,best_r,height,360)
  planet = repair(planet)
  sun_r = best_r*sun_rat
  ring_r = best_r*ring_rat
  orbit_r=ring_r-best_r

  sun=trimesh.creation.annulus(r_min=id/2 , r_max=orbit_r, height=height)

  sun.apply_translation([0, 0,height / 2])

  hob = trimesh.creation.annulus(r_min=orbit_r , r_max=od/2, height=1.5*height)
  hob.apply_translation([0, 0,height / 2])
  
  cutter_list = [hob]
  angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)

  for angle in angles:
      cutter = planet.copy()

      T = translation_matrix([orbit_r * np.cos(angle),orbit_r * np.sin(angle), 0])
      R = rotation_matrix(angle *sun_r / best_r, [0, 0, 1])
      cutter.apply_transform(T @ R)
      cutter_list.append(cutter)

 
  all_cutters = trimesh.boolean.union(cutter_list, engine=bool_engine)
  all_cutters = repair(all_cutters)
  all_cutters.export('hob_sun.stl')
  print([sun.is_volume, all_cutters.is_volume])

  sun = trimesh.boolean.difference([sun, all_cutters], engine=bool_engine)
  sun = repair(sun)
  sun.export("sun2.stl")
  print("sun")
  
  
  ring = trimesh.creation.annulus(r_min=ring_r-best_r , r_max=od/2, height=height)
  ring.apply_translation([0, 0,height / 2])

  
  hob = trimesh.creation.cylinder(radius= ring_r-best_r/N_p, height=height*1.5)
  hob.apply_translation([0, 0,height / 2])
  

  cutter_list = [hob]
  angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)

  for angle in angles:
      cutter = planet.copy()

      T = translation_matrix([orbit_r * np.cos(angle),orbit_r * np.sin(angle), 0])
      R = rotation_matrix(-angle *ring_r / best_r, [0, 0, 1])
      cutter.apply_transform(T @ R)
      cutter_list.append(cutter)

 
  all_cutters = trimesh.boolean.union(cutter_list, engine=bool_engine)
  all_cutters = repair(all_cutters)
  all_cutters.export('hob_ring.stl')
  print([ring.is_volume, all_cutters.is_volume])

  #all_cutters = all_cutters.process(validate=True)

  ring = trimesh.boolean.difference([ring, all_cutters], engine=bool_engine)
  ring = repair(ring)

  #print("Final sun is_volume:", sun.is_volume)
  ring.export("ring.stl")
  
  bearing = ring + sun
  angles = np.linspace(0, 2 * np.pi, num_p, endpoint=False)
  hole = trimesh.creation.cylinder(radius=.6*best_r , height=height)

  hole.apply_translation([0, 0,height / 2])
  planet_out = planet_out.difference(hole,engine=bool_engine)
  planet_out.export('planet.stl')
  for angle in angles:
    planet2=planet_out.copy()
    
    T = translation_matrix([orbit_r * np.cos(angle),orbit_r * np.sin(angle), 0])

    R = rotation_matrix(-angle *ring_r / best_r, [0, 0, 1])

    planet2.apply_transform(T @ R)
    bearing = bearing+planet2

  bearing.export('bearing.stl') 
  
generate_plantery(id=2,od=4,height=.5)