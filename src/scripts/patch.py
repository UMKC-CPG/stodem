import numpy as np


class Patch():

    def __init__(self, settings, i, j, x_num_patches, y_num_patches,
            num_zone_types, zone_types):

        # Initialize the properties of this patch.
        self.zone_index = []
        self.x_location = i
        self.y_location = j

        # Determine the properties of this patch.

        # Initialize the number of citizens on this patch.
        self.num_citizens = int(
                settings.infile_dict[1]["patch"]["initial_num_citizens"])

        # Compute the set of zone index numbers for this patch. All patches of
        #   a given zone and zone type will have the same index number. Index
        #   numbers are computed contiguously from 0, and increasing by 1 for
        #   each "zone-sized" step in the x-direction and increasing by the
        #   number of zones in a row for each row in the
        #   y-direction.
        for zone_type in range(num_zone_types):
            #print("i,j,zone_type =",i,j,zone_type)
            zone_idx = (i // zone_types[zone_type]["x_num_patches"]) + \
                       (j // zone_types[zone_type]["y_num_patches"]) * \
                       (x_num_patches // zone_types[zone_type]["x_num_patches"])
            #print("zone_idx =",zone_idx)
            self.zone_index.append( \
                    (i // zone_types[zone_type]["x_num_patches"]) + \
                    (j // zone_types[zone_type]["y_num_patches"]) * \
                    (x_num_patches // zone_types[zone_type]["x_num_patches"]))


    def sprout_citizens(self, start_index, num_citizens):

        # Add citizens to this patch.
        self.citizen_list = np.array(range(start_index,
                start_index + num_citizens))
