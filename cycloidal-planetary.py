from stl import mesh
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from types import SimpleNamespace


def generate_cycloid(N=6, base_r=0.5, segments=360):
    r = base_r / (2 * N)
    th = np.linspace(0, np.pi/N, 360)
    epi = (base_r + r)*np.column_stack([np.cos(th), np.sin(th)]) - r*np.column_stack([np.cos(th*(base_r + r)/r), np.sin(th*(base_r + r)/r)])
    hyp = (base_r - r)*np.column_stack([np.cos(th + np.pi/N), np.sin(th + np.pi/N)]) + r*np.column_stack([np.cos((base_r - r)*(th + np.pi/N)/r), -np.sin((base_r - r)*(th + np.pi/N)/r)])
    tooth = np.concatenate([epi, hyp])

    phases = np.linspace(0, 2*np.pi, N, endpoint=False)
    rot = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    points = np.concatenate([tooth @ rot(p).T for p in phases])

    dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    uniform_s = np.linspace(0, dist[-1], segments + 1)
    fx = interp1d(dist, points[:,0])
    fy = interp1d(dist, points[:,1])
    
    return SimpleNamespace(
        x=fx(uniform_s), y=fy(uniform_s), r=base_r, N=N, h=2*r
      )

def smooth_path(x,y,N=100):
  distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
  cumul_dist = np.concatenate(([0], np.cumsum(distances)))
  u = cumul_dist / cumul_dist[-1]
  u_fine = np.linspace(0, 1, N)
  interp_x = interp1d(u, x, kind='cubic')
  interp_y = interp1d(u, y, kind='cubic')
  x_smooth = interp_x(u_fine)
  y_smooth = interp_y(u_fine)
  return x_smooth,y_smooth
    
def rot2(x,y,r,th_body,th_spatial):
  x,y = rot(x,y,th_body)
  x,y = rot(x+r,y,th_spatial)
  return x,y
  
def rot(x,y,th):
  return x*np.cos(th)-y*np.sin(th),x*np.sin(th)+y*np.cos(th)
  
def cart(r, theta):
    return r * np.cos(theta), r * np.sin(theta)

def polar(x, y):
    return np.sqrt(x**2 + y**2), np.arctan2(y, x)


def generate_helix(gear, height=1, helix=np.radians(35), radius=None,reverse=False):
    profile = np.column_stack((gear.x, gear.y))
    n_layers = 51
    dz = height / (n_layers - 1)

    # Generate helical layers
    layers = []
    k = 8
    #thetas = 2 * helix  * np.log(np.cosh(k * np.linspace(-1, 1, n_layers))) / k
    thetas = .5 * helix  * (np.cos(np.linspace(0, 2*np.pi, n_layers)))
    for i in range(n_layers):
        th = thetas[i]
        transformed = np.column_stack((
            profile[:, 0]*np.cos(th) - profile[:, 1]*np.sin(th),
            profile[:, 0]*np.sin(th) + profile[:, 1]*np.cos(th)
        ))
        z = np.full((len(profile), 1), i * dz)
        layer = np.hstack((transformed, z))  # [x, y, z]
        layers.append(layer)
    vertices = np.vstack(layers)

    n_points = len(profile)

    # Create cylinder circles at bottom and top
    if radius is None:
        radius = np.max(np.linalg.norm(profile, axis=1)) * 1.1  # Slightly outside if not given
    helix_rad=np.radians(helix)
    # Align bottom circle with first layer rotation
    th_start = thetas[0]
    th_end = thetas[-1]

    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    circle = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))

# Rotate bottom and top circles to align with helical twist
    circle_bottom = np.column_stack((
      circle[:, 0]*np.cos(th_start) - circle[:, 1]*np.sin(th_start),
      circle[:, 0]*np.sin(th_start) + circle[:, 1]*np.cos(th_start)
    ))
    circle_top = np.column_stack((
      circle[:, 0]*np.cos(th_end) - circle[:, 1]*np.sin(th_end),
      circle[:, 0]*np.sin(th_end) + circle[:, 1]*np.cos(th_end)
    ))
    bottom_circle = np.hstack((circle_bottom, np.zeros((n_points, 1))))
    top_circle = np.hstack((circle_top, np.full((n_points, 1), dz*(n_layers-1))))
    vertices = np.vstack((vertices, bottom_circle, top_circle))

    # Create faces
    faces = []

    # Side faces of helical layers
    for i in range(n_layers - 1):
        for j in range(n_points):
            jp = (j + 1) % n_points
            a1 = i * n_points + j
            a2 = i * n_points + jp
            b1 = (i + 1) * n_points + j
            b2 = (i + 1) * n_points + jp
            faces.append([a1, b1, b2])
            faces.append([a1, b2, a2])

    # Connect bottom circle to bottom profile
    bottom_profile_start = 0
    bottom_circle_start = n_layers * n_points
    for j in range(n_points):
        jp = (j + 1) % n_points
        p1 = bottom_profile_start + j
        p2 = bottom_profile_start + jp
        c1 = bottom_circle_start + j
        c2 = bottom_circle_start + jp
        faces.append([p1, c1, c2])
        faces.append([p1, c2, p2])

    # Connect top profile to top circle
    top_profile_start = (n_layers - 1) * n_points
    top_circle_start = bottom_circle_start + n_points
    for j in range(n_points):
        jp = (j + 1) % n_points
        p1 = top_profile_start + j
        p2 = top_profile_start + jp
        c1 = top_circle_start + j
        c2 = top_circle_start + jp
        faces.append([p1, p2, c2])
        faces.append([p1, c2, c1])

    # Connect bottom circle to top circle (cylinder wall)
    for j in range(n_points):
        jp = (j + 1) % n_points
        b1 = bottom_circle_start + j
        b2 = bottom_circle_start + jp
        t1 = top_circle_start + j
        t2 = top_circle_start + jp
        faces.append([b1, b2, t2])
        faces.append([b1, t2, t1])

    return vertices, np.array(faces)
    
def transform_xy2(vertices, r=0,th1=0, th2=0):
    R1 = np.array([
        [np.cos(th1), -np.sin(th1)],
        [np.sin(th1),  np.cos(th1)]
    ])

    R2 = np.array([
        [np.cos(th2), -np.sin(th2)],
        [np.sin(th2),  np.cos(th2)]
    ])

    xy = vertices[:, :2] @ R1.T
    xy += np.array([r,0])
    xy = xy @ R2.T
    transformed = np.hstack((xy, vertices[:, 2][:, None]))
    return transformed
    
def save_stl(filename, vertices, faces):
    stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = vertices[f[j], :]
    stl_mesh.save(filename)
    print(f"saved {filename}")

def generate_conj(planet, ratio, internal=False,clearance=.005):
    base_r = planet.r
    r = base_r * ratio
    circle = np.linspace(0, 2 * np.pi, 1080)

    path = []
    orbit_r = r - base_r if internal else r + base_r
    for th in circle:
        gear_spin = -th*ratio if internal else th*ratio
        xp, yp = rot2(planet.x, planet.y, orbit_r, gear_spin, th)
        rp, tp = polar(xp, yp)
        tp = np.mod(tp, 2*np.pi)
        for r_val, t_val in zip(rp, tp):
            if internal and r_val > r-planet.h*1.5:
              path.append((r_val, t_val))
            elif r_val < r+planet.h*1.5:
              path.append((r_val, t_val))
            else:
              continue
              
    path = np.array(path)
    idx = np.argsort(path[:, 1])
    path_sorted = path[idx]
    r_sorted = path_sorted[:,0]
    t_sorted = path_sorted[:,1]
    dtheta = np.diff(circle)[0]

    r_vals = np.full_like(circle, 0)

    for i, th in enumerate(circle):
      th_min = (th - dtheta/2) % (2*np.pi)
      th_max = (th + dtheta/2) % (2*np.pi)

      if th_min < th_max:
        # No wrapping around 0
        start = np.searchsorted(t_sorted, th_min, side='left')
        end = np.searchsorted(t_sorted, th_max, side='right')
        candidates = r_sorted[start:end]
      else:
        start1 = np.searchsorted(t_sorted, 0, side='left')
        end1 = np.searchsorted(t_sorted, th_max, side='right')
        start2 = np.searchsorted(t_sorted, th_min, side='left')
        end2 = len(t_sorted)
        candidates = np.concatenate((r_sorted[start1:end1], r_sorted[start2:end2]))

      if candidates.size > 0:
        r_vals[i] = np.max(candidates) + clearance if internal else np.min(candidates) - clearance
    # Convert to numpy arrays
    x,y = cart(r_vals,circle)
    x_smooth , y_smooth = smooth_path(x,y,360)
    return SimpleNamespace(
        x=x, y=y, r=r)
 
def generate_plantery(id=2.5,od=4,height=.75,helix=30,save='cycloid_bearing'):
  
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
    for N_p in [6,5,7,8]:
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
  
  planet = generate_cycloid(N_p,best_r,180)
  ring = generate_conj(planet,ring_rat,True)
  sun = generate_conj(planet,sun_rat,False)
  
  theta_scale = np.deg2rad(helix)
    
  planet_th = np.linspace(0,2*np.pi,num_p,endpoint=False)
    
  vertices_p, faces_p = generate_helix(planet, height, theta_scale, best_r/2)
  vertices_s, faces_s = generate_helix(sun, height, -theta_scale/sun_rat, id/2)
  vertices_r, faces_r = generate_helix(ring, height, theta_scale/ring_rat, od/2)
  all_vertices = []
  all_faces = []
  vertex_offset = 0
  save_stl('planet.stl', vertices_p, faces_p)
  save_stl('sun.stl', vertices_s, faces_s)
  save_stl('ring.stl', vertices_r, faces_r)
  
  all_vertices.append(vertices_s)
  all_faces.append(faces_s)
  vertex_offset += vertices_s.shape[0]
  all_vertices.append(vertices_r)
  all_faces.append(faces_r+vertex_offset)
  vertex_offset += vertices_r.shape[0]
    
    
  for th in planet_th:
      v_trans = transform_xy2(vertices_p, ring.r - planet.r, -th*ring_rat, th)
      all_vertices.append(v_trans)
      all_faces.append(faces_p + vertex_offset)
      vertex_offset += v_trans.shape[0]
    
    
  combined_vertices = np.vstack(all_vertices)
  combined_faces = np.vstack(all_faces)
  save_stl(save+'.stl', combined_vertices, combined_faces)
  
  fig,ax = plt.subplots(figsize=(12, 12))
  ax.set_aspect('equal')
  ax.plot(ring.x,ring.y)
  ax.plot(sun.x,sun.y)

  for th in planet_th:
    xp, yp = rot2(planet.x, planet.y, ring.r - planet.r, -th*ring_rat, th)
    ax.plot(xp,yp)
    id_circle = plt.Circle((0,0),id/2,fill=False)
    od_circle = plt.Circle((0,0),od/2,fill=False)
    ax.add_patch(id_circle)
    ax.add_patch(od_circle)
    plt.savefig(save+'.png', dpi=300)
  return
    
generate_plantery(id=2,od=4,height=.75,helix=45,save='demo')
