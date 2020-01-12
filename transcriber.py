#!python3

"""
This is a simple GUI application used to help me improve the transcription
from the GCloud service. It has very limited text-editing features.

How to use:
1. Run the program by Python 3.
2. Click on a cell in "r_txt" to edit the content. You may use Cmd+Z to undo
any edit. But there is no way to revert to the original copy now.
3. Click on Play to play associated audio files. Click on Pause to pause
playing. The timeline and seeking are not supported now.
4. Press Ctrl-Return to save the transcript.
"""

import numpy as np
import pandas as pd
import tkinter as tk
import math
import pyaudio
import wave
from functools import partial
from pathlib import Path


class DataModel:
    """
    Used to hold the input spreadsheet data frame.
    """

    def __init__(self):
        self.df_data = None

    def get_data_v0(self, only_p1=True):
        df_data = pd.read_csv('data.v0.csv')

        # Update the txt_r
        def _get_txt(file_path):
            p = Path(file_path)
            rt = None
            if p.exists():
                with open(p, 'r') as fin:
                    rt = fin.read()
            return rt
        df_data['txt_r'] = df_data['txt_r_path'].apply(_get_txt)

        if only_p1:
            df_data = df_data.query('role == "P1"').copy()
        df_ss = df_data[['audio_path', 'timestamp', 'txt', 'txt_r']]

        self.df_data = df_data
        return df_ss

    def save_txt_entry(self, df_index, new_txt):
        txt_path = self.df_data.loc[df_index, 'txt_r_path']

        with open(txt_path, 'w') as fout:
            fout.write(new_txt)
        print("A new text was saved to {}:\n{}".format(txt_path, new_txt))

    def get_test_dataset(self):
        n_rows, n_cols = 50, 4
        data_shape = (n_rows, n_cols)
        mt_data = np.random.normal(size=data_shape)
        df_data = pd.DataFrame(
            mt_data, columns=['audio_path', 'timestamp', 'txt', 'txt_r'])

        self.df_data = df_data

        return df_data


class AudioPlayer:
    """
    Used to play wav audio files
    """

    def __init__(self):
        self.stream = None
        self.wf = None
        self.p = None

    def play(self, filename):
        """
        Play a wav file
        """
        # Check if the file exists
        if not Path(filename).exists():
            print("[ERR] File not found: {}".format(filename))
            return

        # If there's an existing file playing, stop and cleanup
        if self.stream is not None:
            self.stop()
            self.cleanup()

        self.wf = wf = wave.open(filename, 'rb')
        self.p = p = pyaudio.PyAudio()

        def callback(in_data, frame_count, time_info, status):
            data = wf.readframes(frame_count)
            return data, pyaudio.paContinue

        self.stream = stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(), rate=wf.getframerate(),
            output=True, stream_callback=callback)

        stream.start_stream()

    def stop(self):
        """
        Stop playing a file
        """
        if self.stream.is_active():
            self.stream.stop_stream()

    def pause_or_resume(self):
        if self.stream.is_stopped():
            self.stream.start_stream()
        elif self.stream.is_active():
            self.stream.stop_stream()

    def cleanup(self):
        self.stream.close()
        self.wf.close()
        self.p.terminate()
        self.stream = self.wf = self.p = None


class ScrollFrame(tk.Frame):
    """
    A scrollable frame for TK. The code is adapted from [1].

    References:
        [1] https://gist.github.com/mp035/9f2027c3ef9172264532fcd6262f3b01
    """

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master

        # The base canvas
        self.canvas = tk.Canvas(self, borderwidth=0, background="#ffffff")
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Place a viewport
        self.view_port = tk.Frame(self.canvas, background="#ffffff")
        # Place a scrollbar
        self.v_scrollbar = tk.Scrollbar(self, orient="vertical",
                                        command=self.canvas.yview)
        # Attach the scrollbar action to the scroll of the canvas
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set)
        # Pack the scrollbar to the right of self
        self.v_scrollbar.pack(side="right", fill="y")
        # Pack canvas to left of self and expand to fil
        self.canvas.pack(side="left", fill="both", expand=True)

        # Add the view port frame to the canvas
        self.canvas_window = self.canvas.create_window(
            (1, 1), window=self.view_port, anchor="nw", tags="self.view_port")

        # Bind an event whenever the size of the viewPort frame changes.
        self.view_port.bind("<Configure>", self.on_frame_configured)
        # Bind an event whenever the size of the viewPort frame changes.
        self.canvas.bind("<Configure>", self.on_canvas_configured)

        # perform an initial stretch on render, otherwise the scroll region has
        # a tiny border until the first resize
        self.on_frame_configured(None)

        self.pack(side="top", fill="both", expand=True)

    def get_viewport(self):
        return self.view_port

    def scroll_to_top(self):
        self.canvas.yview_moveto(0.0)

    def on_frame_configured(self, event):
        """
        Reset the scroll region to encompass the inner frame
        """
        # Whenever the size of the frame changes, alter the scroll region
        # respectively.
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configured(self, event):
        """
        Reset the canvas window to encompass inner frame when required
        """
        canvas_width = event.width
        # Whenever the size of the canvas changes alter the window region r
        # espectively.
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)

    def _on_mousewheel(self, event):
        """
        References:
            - https://stackoverflow.com/questions/17355902/python-tkinter-binding-mousewheel-to-scrollbar
        """
        self.canvas.yview_scroll(-1 * int(event.delta), "units")


class TranscriberMainView(tk.Frame):
    def __init__(self, data_model, n_row_pp=8, master=None):
        super().__init__(master)
        self.master = master
        self.pack()

        # Init

        self.data_model = data_model
        self.df_data = None
        self.n_row = None
        self.n_row_pp = n_row_pp
        self.total_pages = None
        self.cur_page = None

        self.audio_player = AudioPlayer()

        # Add a control panel

        ctrl_frame = tk.Frame(master)
        ctrl_frame.pack(side="bottom", fill="x", expand=True)

        prev_btn = tk.Button(ctrl_frame, text="< PREV")
        prev_btn.config(command=self.goto_prev_page)
        next_btn = tk.Button(ctrl_frame, text="NEXT >")
        next_btn.config(command=self.goto_next_page)

        self.pager_page_var = pager_page_var = tk.StringVar()
        self.pager_page_entry = pager_page_entry = tk.Entry(
            ctrl_frame, textvariable=pager_page_var)

        prev_btn.pack(side="left", fill="x", expand=True)
        pager_page_entry.pack(side="left", fill="x", expand=True)
        next_btn.pack(side="left", fill="x", expand=True)

        # Set the init display

        self.load_data(data_model.get_data_v0())
        self.cell_frame = None
        self.refresh_display()

    def load_data(self, df_data):
        """
        Load the input data and initialize the pager
        """
        self.df_data = df_data
        self.n_row = n_row = df_data.shape[0]
        self.total_pages = math.ceil(n_row / self.n_row_pp)
        self.cur_page = 0

    def goto_next_page(self):
        """
        Go to the next page
        """
        next_page = self.cur_page + 1
        if next_page <= self.total_pages:
            self.cur_page = next_page
            self.refresh_display()

    def goto_prev_page(self):
        """
        Go to the previous page
        """
        next_page = self.cur_page - 1
        if next_page >= 0:
            self.cur_page = next_page
            self.refresh_display()

    def update_pager_counter(self):
        """
        Update the pager's counter
        """
        self.pager_page_entry.config(state=tk.NORMAL)
        self.pager_page_var.set(
            "{} / {}".format(self.cur_page, self.total_pages))
        self.pager_page_entry.config(state=tk.DISABLED)

    def refresh_display(self):
        """
        Update the current view by the pager
        """
        start_i = self.cur_page * self.n_row_pp
        end_i = start_i + self.n_row_pp
        df_snapshot = self.df_data[start_i:end_i]

        self.update_pager_counter()

        # Destroy the existing view and add a new one
        if self.cell_frame:
            self.cell_frame.destroy()
        self.insert_data(df_snapshot)

    def insert_data(self, df):
        """
        Insert a dataframe into the viewport
        """
        def _txt_onclick_event(event, debug=False):
            widget = event.widget
            widget.config(state=tk.NORMAL)
            df_index = widget.meta_df_index

            if debug:
                print("A text entry was being clicked: {}".format(df_index))

        def _txt_onfocusout_event(event):
            widget = event.widget
            widget.master.focus()
            widget.config(state=tk.DISABLED)

            df_index = widget.meta_df_index
            new_txt = widget.get("1.0", tk.END)
            self.data_model.save_txt_entry(df_index, new_txt)

            # print("A text entry was finished:\n{}: {}".format(df_index, new_txt))

        # A base cell frame
        self.cell_frame = tk.Frame(self)
        self.cell_frame.pack(side="top", fill="both", expand=True)

        # Add column names
        col_blank_head = tk.Label(self.cell_frame)
        col_blank_head.grid(row=0, column=0)
        for j, col_name in enumerate(df.columns):
            col_label = tk.Label(self.cell_frame, text=col_name)
            col_label.grid(row=0, column=(j + 1))

        # Add rows
        for i, row in df.iterrows():
            row_head = tk.Label(self.cell_frame, text=str(i))
            row_head.grid(row=i + 1, column=0)

            # Add columns from each row
            for j, col_val in enumerate(row):

                # A cell content holder
                cell_entry = tk.Text(
                    self.cell_frame, width=32, height=8, borderwidth=4)
                cell_entry.insert("1.0", str(col_val))
                cell_entry.config(state=tk.DISABLED)

                # If the entry is for the user to edit
                if df.columns[j] == "txt_r":
                    cell_entry.config(undo=True, maxundo=5)
                    # Just append my meta attributes to the text entry
                    cell_entry.meta_df_index = i
                    cell_entry.bind("<Button-1>", _txt_onclick_event)
                    cell_entry.bind("<Control-Return>", _txt_onfocusout_event)

                cell_entry.grid(row=i + 1, column=j + 1)

            # Add a player panel and buttons

            player_panel = tk.Frame(self.cell_frame)

            play_btn = tk.Button(player_panel, text="PLAY")
            # Note: the lambda closure will cache the audio filename
            # This is not a pretty solution in any way but just works.
            play_btn.config(
                command=lambda x=row['audio_path']: self._play_audio_file(x))
            pause_btn = tk.Button(player_panel, text="PAUSE")
            pause_btn.config(command=self._pause_or_resume_audio)
            play_btn.pack(side="left")
            pause_btn.pack(side="left")

            player_panel.grid(row=i + 1, column=j + 2)

    def _play_audio_file(self, filename):
        print("Play " + filename)
        self.audio_player.play(filename)

    def _pause_or_resume_audio(self):
        self.audio_player.pause_or_resume()


class TranscriberApp(tk.Frame):
    """
    The main GUI app frame.
    """

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master

        self.winfo_toplevel().title("Og-ConVis Transcription Helper")

        data_model = DataModel()

        self.scroll_frame = ScrollFrame(master=self)
        self.transcriber_frame = TranscriberMainView(
            data_model, n_row_pp=8, master=self.scroll_frame.view_port)

        self.pack(expand=True, fill="both")


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1024x720+0+0")
    app = TranscriberApp(master=root)
    app.mainloop()
