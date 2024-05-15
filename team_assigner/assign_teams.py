import os
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.cluster import KMeans

class TeamAssigner:
    """
    A class used to assign team colors to players in video frames using K-Means clustering.

    Attributes:
    -----------
    team_colors : dict
        A dictionary storing the team colors.
    player_team_dict : dict
        A dictionary mapping player IDs to team IDs.
    """

    def __init__(self) -> None:
        """
        Initializes the TeamAssigner class with empty dictionaries for team colors and player-team mapping.
        """

        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        """
        Creates and fits a K-Means clustering model to the given image.

        Parameters:
        -----------
        image : numpy.ndarray
            The image to be clustered.

        Returns:
        --------
        KMeans
            A fitted K-Means clustering model with 2 clusters.
        """

    def get_clustering_model(self,image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans


    def get_player_color(self, frame, bbox):
        """
        Determines the color of a player within the given bounding box in the frame.

        Parameters:
        -----------
        frame : numpy.ndarray
            The video frame (image) containing the player.
        bbox : list
            The bounding box coordinates [x1, y1, x2, y2] of the player.

        Returns:
        --------
        numpy.ndarray
            The RGB color of the player.
        """

    def get_player_color(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        top_half_image = image[0:int(image.shape[0]/2),:]

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels forr each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color


    def assign_team_color(self,frame, player_detections):
        """
        Assigns team colors based on the player detections in the given frame.

        Parameters:
        -----------
        frame : numpy.ndarray
            The video frame (image) containing the players.
        player_detections : dict
            A dictionary containing player detections with bounding boxes.

        Returns:
        --------
        None
        """

        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color =  self.get_player_color(frame,bbox)
            player_colors.append(player_color)
        
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self,frame,player_bbox,player_id):
        """
        Determines the team ID for a player based on their color and assigns it if not already assigned.

        Parameters:
        -----------
        frame : numpy.ndarray
            The video frame (image) containing the player.
        player_bbox : list
            The bounding box coordinates [x1, y1, x2, y2] of the player.
        player_id : int
            The unique ID of the player.

        Returns:
        --------
        int
            The team ID of the player.
        """

        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame,player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id+=1

        if player_id ==91:
            team_id=1

        self.player_team_dict[player_id] = team_id

        return team_id