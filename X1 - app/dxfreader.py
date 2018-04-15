#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 21:56:10 2018

@author: cricket
"""

import math
import numpy as np
from shapely.geometry.point import Point
from shapely.geometry import Polygon, LineString
import dxfgrabber as dxfgb
import os


class DxfParser():
    def __init__(self, edge_color=[ 3, 7], #List of possible colors for cuted edges
                 etching_color=2,
                 min_edge_length=.5, #Maximum length for ARCS and ELLIPSE
                 roundigexp=3, #Number of digits to keep
                 max_distance_correction=.75, #Maximum distance to close open loops
                 ):
        self.min_edge_length = min_edge_length
        self.etching_color = etching_color
        self.edge_color = edge_color
        self.roundigexp = roundigexp
        self.details = {}
        self.color = 0
        self.max_distance_correction = max_distance_correction

    def load_dxf(self, dxfpath):
        """Loads the DXF file"""
        self.dxf = dxfgb.readfile(dxfpath)

        #Most common color calculation
        colors = [element.color for element in self.dxf.modelspace() if element.linetype == 'CONTINUOUS'and element.dxftype != 'TEXT' and element.layer != 'SURFACES' and element.color in self.edge_color]
        self.color = np.argmax(np.bincount(colors))
        #print('color ID', self.color)

        # Note reading
        notes = [element for element in self.dxf.modelspace() if element.layer == 'NOTES' and element.dxftype == 'TEXT']

        bend_angle = []
        bend_direction = []
        bend_radius = []
        bend_center = []
        self.details['radius_approx'] = False

        for note in notes: #We must first find the thickness
            rawnote = note.text
            coord = note.insert
            # lists all the notes on the same line after the current note
            possiblevalue = [x for x in notes if coord[1]==x.insert[1] and coord[0]<x.insert[0]]
            distances = [Point(x.insert).distance(Point(coord)) for x in possiblevalue]
            index = np.argsort(distances)# if len(distances) !=0 else None

            if rawnote.startswith('THICKNESS:'): #It's the thickness
                try:
                    self.details['thickness'] = float(possiblevalue[index[0]].text.replace(',', '.'))
                except:
                    self.details['thickness'] = np.nan

            elif rawnote.startswith('UNITES:'):#It's the unit
                self.details['unit'] = possiblevalue[index[0]].text

            elif rawnote.startswith('MATIERE'): #It's the material
                self.details['material'] = possiblevalue[index[0]].text

            elif rawnote.startswith('RAYON'): #It's the general radius
                try:
                    self.details['radius'] = float(possiblevalue[index].text)
                except:
                    self.details['radius'] = np.nan

        for note in notes: #Specific loop for bends
            rawnote = note.text
            coord = note.insert

            if rawnote == 'UP' or rawnote == 'DOWN': #It's a bend !
                x,y = coord
                if rawnote == 'UP':
                    coordtext = (x-6.25,y-4.25)
                elif rawnote == 'DOWN':
                    coordtext = (x,y-1.5)
                else:
                    coordtext = (x-6.25,y)

                try: #Notes with bend radius
                    # lists all the notes on the same line AFTER the current note
                    possiblevalue = [x for x in notes if coord[1]==x.insert[1] and coord[0]<x.insert[0]]
                    distances = [Point(x.insert).distance(Point(coord)) for x in possiblevalue]
                    index = np.argsort(distances)

                    angle = float(possiblevalue[index[0]].text)
                    radius = float(possiblevalue[index[3]].text)

                except: #Notes without bend radius
                    # lists all the notes on the same line BEFORE the current note
                    possiblevalue = [x for x in notes if coord[1]==x.insert[1] and coord[0]>x.insert[0]]
                    distances = [Point(x.insert).distance(Point(coord)) for x in possiblevalue]
                    index = np.argsort(distances)

                    angle = float(possiblevalue[index[1]].text)
                    radius = self.details['thickness']

                self.details['radius_approx'] = True
                bend_direction.append(rawnote)
                bend_angle.append(angle)
                bend_radius.append(radius)
                bend_center.append(coordtext)

        self.details['bend_angle'] = bend_angle
        self.details['bend_direction'] = bend_direction
        self.details['bend_radius'] = bend_radius
        self.details['bend_center'] = bend_center

    def _bend_placement(self, limit_distance=3):
        """Populates information such as nuber of bends, direction"""
        axis = []
        tangents_coords = []
        
        #List of axis creation
        tangents = [(element.start, element.end) for element in self.dxf.modelspace() if element.linetype == 'PHANTOM']
        tangentsangles = [np.round(self.angleline(LineString(tangentcoord)), self.roundigexp) for tangentcoord in tangents]
        #print(tangentsangles)
        possible_axis = [element for element in self.dxf.modelspace() if element.linetype == 'CENTER' and element.dxftype == 'LINE' ]
        #centeraxis = [list(LineString((x.start, x.end)).interpolate(0.5, normalized=True).coords)[0] for x in possible_axis]
        
        #Candidate bend lines
        possible_bendline = [element for element in possible_axis if np.round(self.angleline(LineString((element.start, element.end))), self.roundigexp) in tangentsangles]
        
        #Conversion to coordinates
        for lines in possible_bendline:
            segment = np.round((lines.start, lines.end), decimals=self.roundigexp)[:,0:2]
            axis.append(tuple(map(tuple, segment)))
        
        bend_index = []
        
        for angle, radius, center in zip(self.details['bend_angle'], self.details['bend_radius'], self.details['bend_center']):
            #L= angle*R+(0.1594*ln(R/T)+0.51722)*T   
            thickness = self.details['thickness']
            #Flat lengthcenteraxis
            flatdistance = np.radians(angle)*radius+(0.1594*math.log(radius/thickness,math.exp(1))+0.51722)*thickness
            
            #select the axes the n-best axes according the note
            axisdist = [(LineString(x).distance(Point(center)))**2 for x in axis]
            sort_index = np.argsort(axisdist)
            
            #Searching of the tangent line associated with te bend
            for indexposition in sort_index:
                candidate_axis = axis[indexposition]
                
                angle_candidate = np.round(self.angleline(LineString(candidate_axis)), self.roundigexp)
                tangents_candidates = [x for x in tangents if angle_candidate==np.round(self.angleline(LineString(x)), self.roundigexp)]
                distances = [(LineString(candidate_axis).distance(LineString(x).interpolate(0.5, normalized=True))-flatdistance/2)**2 for x in tangents_candidates]
                #print(indexposition, distances, candidate_axis, len(tangents_candidates), angle_candidate)
                mask = np.array(distances)<limit_distance
                if np.any(mask):
                    bend_index.append(indexposition)
                    tangents_coords.append(np.array(tangents_candidates)[mask])
                    break
            else:
                #print('NOK')
                bend_index.append(sort_index[0])
                tangents_coords.append([])
                


        self.details['bend_line'] = [axis[k] for k in bend_index]
        self.details['tangents_coords'] = tangents_coords
        #self.details['bend_center'] = [centeraxis[k] for k in bend_index]
        self.details['punch_length'] = [LineString(x).length for x in self.details['bend_line'] ]
        
        self.details['deformation_length'] = [np.array(np.array([LineString(coord).length for coord in tangentset]).mean()).sum() for tangentset in self.details['tangents_coords']]

    def _pattern_details(self, color):
        """Constructs the pattern with shapely"""
        patterns = []
        closedpatterns = []
        toparse = [element for element in self.dxf.modelspace() if element.color == self.color and element.linetype == 'CONTINUOUS']
        validentities = ['LINE', 'ARC', 'ELLIPSE', 'CIRCLE', 'SPLINE', 'LWPOLYLINE' ]

        for element in [x for x in toparse if x.dxftype in validentities]:
            #print(element)
            if element.dxftype == 'LINE':
                coordsectlist = np.round(np.array((element.start, element.end)), decimals=self.roundigexp)
            elif element.dxftype == 'ARC':
                segment = self._arccoord(element.center, element.radius, element.start_angle, element.end_angle)
                coordsectlist = np.round(segment, decimals=self.roundigexp)
            elif element.dxftype == 'ELLIPSE':
                segment = self._ellipsecoord(element.center, element.major_axis, element.ratio, element.start_param, element.end_param)
                coordsectlist = np.round(segment, decimals=self.roundigexp)
            elif element.dxftype == 'CIRCLE':
                center = Point([round(x, self.roundigexp) for x in element.center])
                circle = center.buffer(element.radius)
                coordsectlist = np.round(circle.exterior.coords, self.roundigexp)
            elif element.dxftype == 'SPLINE':
                coordsectlist = np.round(element.control_points, self.roundigexp)[:, 0:2]
            elif element.dxftype == 'LWPOLYLINE':
                coordsectlist = np.round(element.points, self.roundigexp)

            #List of segment construction
            if element.dxftype not in ['CIRCLE', ]:
                patterns.append(coordsectlist)
            else:
                closedpatterns.append(coordsectlist)

        # First round with no forgivness
        closedpatterns, openpatterns = self._looping_calc(closedpatterns, patterns, limit=0)

        # Second round with some forgivness if needed
        if len(openpatterns) > 1:
            print(len(openpatterns))
            print('round 2 !')
            closedpatterns, openpatterns = self._looping_calc(closedpatterns, openpatterns, limit=self.max_distance_correction)

        return (closedpatterns, openpatterns)

    def _looping_calc(self, closedpatterns, patterns, limit=0):
        """Routine used to create closed loops"""
        openpatterns = []
        # Construcion of the different patterns
        currentsegment = np.array(patterns[0])
        del patterns[0]
        looping = True

        while looping:
            initloop = False
            modifiedloop = False
            for index, segment in enumerate(patterns):

                start = tuple(np.array(segment[0]).tolist())
                end = tuple(np.array(segment[-1]).tolist())
                startsegment = tuple(np.round(currentsegment[0], self.roundigexp).tolist())
                endsegment = tuple(np.round(currentsegment[-1], self.roundigexp).tolist())

                # Attachment 1?
                if LineString((start, endsegment)).length <= limit:
                    currentsegment = np.concatenate((currentsegment, segment[::]))
                    del patterns[index]
                    modifiedloop = True
                    break
                # Attachment 2?
                elif LineString((end, endsegment)).length <= limit:
                    currentsegment = np.concatenate((currentsegment, segment[:: -1]))
                    del patterns[index]
                    modifiedloop = True
                    break
                # Attachment 3?
                elif LineString((startsegment, start)).length <= limit:
                    currentsegment = np.concatenate((segment[::-1], currentsegment))
                    del patterns[index]
                    modifiedloop = True
                    break
                # Attachment 4?
                elif LineString((startsegment, end)).length <= limit:
                    currentsegment = np.concatenate((segment[::], currentsegment))
                    del patterns[index]
                    modifiedloop = True
                    break

            # Is it actually closed?
            if LineString((tuple(currentsegment[0]), tuple(currentsegment[-1]))).length <= limit:
                closedpatterns.append(currentsegment)
                # print('closed loop')
                initloop = True
                #modifiedloop = True

            # The loop is not closed and we exhausted all the options
            if not modifiedloop:
                openpatterns.append(currentsegment)
                #print('Possible PB', openpatterns)
                initloop = True
                if 'possible_imperfection' not in self.details:
                    self.details['possible_imperfection'] = []
                temp = self.details['possible_imperfection']
                temp.append((startsegment, endsegment))
                self.details['possible_imperfection'] = temp

            # Finished a segment and need to initiate a new one
            if initloop:
                if (len(patterns) == 1) and not modifiedloop:
                    looping = False
                    openpatterns.append(currentsegment)
                    openpatterns.append(patterns[0])
                elif len(patterns) == 0:
                    looping = False
                else:
                    currentsegment = np.array(patterns[0]) 
                    del patterns[0]
                    initloop = False


        return (closedpatterns, openpatterns)

    def _surface(self):
        """Misc Surface calculation"""
        areas = np.array([Polygon(poly).area for poly in self.details['closed_patterns']])
        lengths = np.array([Polygon(poly).length for poly in self.details['closed_patterns']])
        biggestarea = np.argmax(areas)
        
        self.details['cut_length'] = lengths.sum()
        self.details['total_area'] = 2 * areas[biggestarea] - areas.sum()
        
        self.mainpattern = Polygon(self.details['closed_patterns'][biggestarea])
        
        self.details['minimum_rectangle_coords'] = self.mainpattern.minimum_rotated_rectangle.exterior.coords[:]
        self.details['minimum_rectangle_area'] = Polygon(self.details['minimum_rectangle_coords']).area
        point1 = Point(self.details['minimum_rectangle_coords'][0])
        point2 = Point(self.details['minimum_rectangle_coords'][1])
        point3 = Point(self.details['minimum_rectangle_coords'][2])
        
        dim1 = point1.distance(point2)
        dim2 = point2.distance(point3)
        self.details['minimum_rectangle_dim1'] = np.array((dim1, dim2)).max()
        self.details['minimum_rectangle_dim2'] = np.array((dim1, dim2)).min()
        
        self.details['no_hole_area'] = self.mainpattern.area
        
        self.details['num_closed_patterns'] = len(self.details['closed_patterns'])
        self.details['num_open_patterns'] = len(self.details['open_patterns'])
        
        self.details['convex_hull_coords'] = self.mainpattern.convex_hull.exterior.coords[:]
        self.details['convex_hull_area'] = Polygon(self.details['convex_hull_coords']).area

    def _arccoord(self, center, radius, start_angle, end_angle):
        """Sub-feature to calculate arc coordiantes"""
        xc, yc = center
        if start_angle > end_angle:
            start_angle -= 360
        cord = self.min_edge_length if radius > self.min_edge_length else radius #Make sure that the ration edge length and ratio is correct
        angle_eq = 2 * np.arcsin(cord / (2*radius)) #Radians
        steps_details = math.floor(np.radians(end_angle - start_angle) / angle_eq)
        steps_number = np.array((2, abs(steps_details))).max()
        toreturn = np.array([])
        for angle in np.linspace(np.radians(start_angle), np.radians(end_angle), steps_number):
            xa = xc + radius * np.cos(angle)
            ya = yc + radius * np.sin(angle)
            toreturn = np.concatenate((toreturn, np.array((xa,ya))))
        #print(center, radius, start_angle, end_angle, steps_number)
        return toreturn.reshape(-1,2)

    def _ellipsecoord(self, center, major_axis, ratio, start_angle, end_angle):
        """Sub-feature to calculate ellipse coordiantes"""
        xc, yc, _ = center
        xe, ye, _ = major_axis
        rmax = np.sqrt(xe**2 + ye**2)
        rmin = rmax * ratio
        radius = (rmin + rmax) / 2

        if xe == 0 and ye > 0:
            angle_ellipse = np.pi / 2
        elif xe == 0 and ye < 0:
            angle_ellipse = -np.pi / 2
        elif ye == 0 and xe > 0:
            angle_ellipse = 0
        elif ye == 0 and xe < 0:
            angle_ellipse = np.pi
        elif xe < 0 and ye < 0:
            angle_ellipse = np.arctan(ye / xe) + np.pi
        elif xe < 0 and ye > 0:
            angle_ellipse = -(np.pi - np.arctan(ye / xe))
        else:
            angle_ellipse = np.arctan(ye / xe)


        if start_angle > end_angle:
            start_angle -= np.pi

        cord = radius if radius < self.min_edge_length else self.min_edge_length #Make sure that the ration edge length and ratio is correct
        angle_eq = 2 * np.arcsin(cord / (2*radius)) #Radians
        steps_number = math.floor((end_angle - start_angle) / angle_eq)
        toreturn = np.array([])
        for angle in np.linspace(start_angle, end_angle, steps_number):
            intx = rmax * np.cos(angle)
            inty = rmin * np.sin(angle)
            xa = xc + intx * np.cos(angle_ellipse) - inty * np.sin(angle_ellipse)
            ya = yc + intx * np.sin(angle_ellipse) + inty * np.cos(angle_ellipse)
            toreturn = np.concatenate((toreturn, np.array((xa, ya))))
        return toreturn.reshape(-1,2)

    def parse(self, dxfpath):
        """Method to call to parse a DXF file, will return a dict with all the needed information"""
        self.details = {}
        self.load_dxf(dxfpath)
        # Cutting patterns
        self.details['closed_patterns'], self.details['open_patterns']= self._pattern_details(self.edge_color)

        # Hetching pattern
        # TO DO

        # surface calculations
        self._surface()
        self._bend_placement()

        self.details['bend_bend_distance'], self.details['bend_bend_angle'], self.details['merged_bend'] = self._bend_bend()
        self.details['bend_edge_distance'], self.details['bend_edge_angle'],  self.details['bend_edge_length']= self._bend_edge()

        return self.details

    def extendline(self, line, length):
        """Extends the linestring ate each each with the selected length"""

        #Line angle calculation
        xs, xe = line.xy[0]
        ys, ye = line.xy[1]

        if xs == xe:
            if ye < ys:
                xe, ye, xs, ys = xs, ys, xe, ye
            return LineString(((xs, ys-length), (xe, ye+length)))
        else:
            if xe > xs:
                xe, ye, xs, ys = xs, ys, xe, ye
            angle = np.arctan((ye-ys)/(xe-xs))
            return LineString(((xs+np.cos(angle)*length, ys+np.sin(angle)*length*np.sign(angle)), (xe-np.cos(angle)*length, ye-np.sin(angle)*length*np.sign(angle))))

    def angleline(self, linestring):
        """Calculates the angle of a line"""
        xs, xe = linestring.xy[0]
        ys, ye = linestring.xy[1]

        if xs == xe:
            angle = np.pi/2
        else:
            angle = np.arctan((ye-ys)/(xe-xs))

        return math.degrees(angle)

    def set_params(self, **kwargs):
        """Used to set all the params"""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_params(self):
        """Method to get all the parameters of the DXF parser"""
        return {key:value for key, value in vars(self).items() }

    def _bend_bend(self):
        """Measures distance and angle between bends"""
        bend_bend_distance, bend_bend_angle, merged_bend = [], [], []
        
        if len(self.details['bend_radius']) < 2:
            #Early break
            return bend_bend_distance, bend_bend_angle, merged_bend
        
        
        
        for index, _ in enumerate(self.details['bend_line']):
            rem_coord = self.details['bend_line'].copy()
            coord = rem_coord.pop(index)
            
            rem_angle = self.details['bend_angle'].copy()
            angle = rem_angle.pop(index)
            
            rem_direction = self.details['bend_direction'].copy()
            direction = rem_direction.pop(index)
            
            rem_radius = self.details['bend_radius'].copy()
            radius = rem_radius.pop(index)
            
            rem_center = self.details['bend_center'].copy()
            center = rem_center.pop(index)
            
            bend_distance = [Point(center).distance(Point(x)) for x in rem_center]
            bend_bend_distance.append(bend_distance)
            
            bend_angle = [self.angleline(LineString(coord)) - self.angleline(LineString(x)) for x in rem_coord]
            bend_bend_angle.append(bend_angle)
            
            extended_cord = self.extendline(LineString(coord), self.details['minimum_rectangle_dim1'])
            comp_distance = [1 if extended_cord.distance(Point(x))<10**(-self.roundigexp) else 0 for x in rem_center]
            #print(comp_distance)
            comp_angle = [1 if x == angle else 0 for x in rem_angle]
            #print(comp_angle)
            comp_direction = [1 if x == direction else 0 for x in rem_direction]
            #print(comp_direction)
            comp_radius = [1 if x == radius else 0 for x in rem_radius]
            #print(comp_radius)
            asm_bend =[1 if a+b+c+d == 4 else 0 for a,b,c,d in zip(comp_distance, comp_angle, comp_direction, comp_radius)]
            merged_bend.append(np.array(asm_bend).sum())
            
            #print(coord, angle, direction, radius, center)
        
        return bend_bend_distance, bend_bend_angle, merged_bend

    def _bend_edge(self, minimal_length=20):
        """Measures distance and angle between bends and outside edges"""
        bend_edge_distance, bend_edge_angle, bend_edge_length = [], [], []
        if len(self.details['bend_radius']) == 0:
            # Early break
            return bend_edge_distance, bend_edge_angle, bend_edge_length

        coordpair = [(cell,self.mainpattern.exterior.coords[index+1]) for index, cell in enumerate(self.mainpattern.exterior.coords[:-1])]
        candidate_edges = [coord for coord in list(coordpair) if LineString(coord).length >= minimal_length]
        for line, center in zip(self.details['bend_line'], self.details['bend_center']):
            bend_edge_distance.append([LineString(line).distance(LineString(x)) for x in candidate_edges])
            bend_edge_angle.append([abs(self.angleline(LineString(line)) - self.angleline(LineString(x))) for x in candidate_edges])
            bend_edge_length.append([LineString(x).length for x in candidate_edges])

        return bend_edge_distance, bend_edge_angle, bend_edge_length

if __name__ == "__main__":
    #List of all the dixs in the folder
    dxffolder = '../Y2 - Sample DXF'
    dxflist = [os.path.join(dxffolder, file) for file in os.listdir(dxffolder) if file.endswith('.dxf')]
    for index, path in enumerate(dxflist):
        print('{:02d} --> '.format(index), path.split('/')[-1])
        pass

    fileid = 11

    dxfparser = DxfParser(min_edge_length=1, roundigexp=2)
    print(dxflist[fileid])
    details = dxfparser.parse(dxflist[fileid])
    print(details.keys())