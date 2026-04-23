from pathlib import Path

import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display as ipy_display

from threshold_activity_classifier.ui.napari_viewer import (
    NapariImageViewer,
    filter_images_by_sample_id_and_z_index,
    _load_optional,
)

from threshold_activity_classifier.utils.df_utils import add_meta_info

import importlib.util
NAPARI_AVAILABLE = importlib.util.find_spec('napari') is not None


class ImageViewer:
    """
    ipywidget-based image viewer that loads from an activity summary CSV
    and opens images in Napari for 2D single-image or 3D z-stack viewing.

    The summary CSV is expected to have columns:
        image, mcherry_path, label_path, activity_path, brightfield_path,
        n_instances, activity_ratio
    and optionally an ``ID`` column for sample grouping (used by 3D view).

    Parameters
    ----------
    file_handler : optional
        Object with an ``extract_unique_id(filename: str) -> str`` method.
        Only used when the loaded CSV does not already contain an ``ID`` column.
    """

    def __init__(self, file_handler=None):
        self.file_handler = file_handler
        self._napari_viewer = NapariImageViewer()
        self._summary_df: pd.DataFrame | None = None
        self._selected_image: list[str | None] = [None]

        self._build_widgets()
        self._wire_events()

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _build_widgets(self):
        # CSV input row
        self._csv_input = widgets.Text(
            description='Summary CSV:',
            placeholder='Path to activity _summary.csv',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='800px'),
        )
        self._load_btn = widgets.Button(
            description='Load CSV',
            button_style='info',
            icon='upload',
            layout=widgets.Layout(width='120px'),
        )
        self._load_out = widgets.Output()

        # View mode toggle
        self._view_mode = widgets.RadioButtons(
            options=[('2D Single Image', '2d'), ('3D Z-Stack (by Sample)', '3d')],
            value='2d',
            description='View Mode:',
            layout=widgets.Layout(width='300px'),
            style={'description_width': '100px'},
        )

        # 2D image list
        self._image_list = widgets.Select(
            options=[],
            description='Images:',
            layout=widgets.Layout(width='600px', height='200px'),
        )

        # 3D sample selector
        self._sample_selector = widgets.Select(
            options=['No samples available'],
            description='Samples:',
            layout=widgets.Layout(width='400px', height='200px'),
            style={'description_width': '80px'},
        )

        # Display option checkboxes
        self._show_mcherry = widgets.Checkbox(
            value=True, description='mCherry', layout=widgets.Layout(width='150px')
        )
        self._show_labels = widgets.Checkbox(
            value=True, description='Instance Labels', layout=widgets.Layout(width='150px')
        )
        self._show_activity = widgets.Checkbox(
            value=True, description='Activity Labels', layout=widgets.Layout(width='150px')
        )
        self._show_bf = widgets.Checkbox(
            value=True, description='Brightfield (BF)', layout=widgets.Layout(width='150px')
        )

        # Info / status outputs
        self._sample_info_out = widgets.Output()
        self._popup_out = widgets.Output()

        # Per-image stats panel (2D only)
        self._image_stats = widgets.HTML(
            value='',
            layout=widgets.Layout(margin='4px 0'),
        )

        # 2D / 3D mode containers
        self._container_2d = widgets.VBox([
            widgets.HTML('<b>Select an image to view:</b>'),
            self._image_list,
            self._image_stats,
        ])
        # z-slice range for 3D view (z0 is always skipped; max_z is configurable)
        self._max_z = widgets.BoundedIntText(
            value=20,
            min=1,
            max=999,
            description='Max z-slice:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='160px'),
        )

        self._container_3d = widgets.VBox([
            widgets.HTML('<b>Select a sample for 3D z-stack view:</b>'),
            self._sample_selector,
            widgets.HBox([
                widgets.HTML('<small>z0 is always excluded &nbsp;|&nbsp; </small>'),
                self._max_z,
            ]),
        ])
        self._container_3d.layout.display = 'none'

        # Open / close buttons
        self._open_btn = widgets.Button(
            description='Open in Napari',
            button_style='primary',
            icon='external-link',
            layout=widgets.Layout(width='200px'),
        )
        self._close_btn = widgets.Button(
            description='Close Viewer',
            button_style='warning',
            icon='times',
            layout=widgets.Layout(width='150px'),
        )
        self._show_multiple_btn = widgets.Button(
            description='Show Sample Images',
            button_style='info',
            icon='images',
            layout=widgets.Layout(width='200px'),
        )

        # Assemble full layout
        self.widget = widgets.VBox([
            widgets.HTML('<h1>Visualize Images in Napari</h1>'),
            widgets.HTML('<b>Activity summary CSV:</b>'),
            widgets.HBox([self._csv_input, self._load_btn]),
            self._load_out,
            widgets.HTML('<hr/>'),
            self._view_mode,
            self._container_2d,
            self._container_3d,
            self._sample_info_out,
            widgets.HTML('<b>Display Options:</b>'),
            widgets.HBox([
                self._show_mcherry, self._show_labels,
                self._show_activity, self._show_bf,
            ]),
            widgets.HTML('<hr/>'),
            widgets.HBox([self._open_btn, self._show_multiple_btn, self._close_btn]),
            self._popup_out,
        ])

    def _wire_events(self):
        self._load_btn.on_click(self._on_load_csv)
        self._view_mode.observe(self._on_view_mode_change, names='value')
        self._image_list.observe(self._on_image_select, names='value')
        self._sample_selector.observe(self._update_sample_info, names='value')
        self._view_mode.observe(lambda _: self._update_sample_info(), names='value')
        self._open_btn.on_click(self._on_open)
        self._show_multiple_btn.on_click(self._on_show_multiple)
        self._close_btn.on_click(self._on_close)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_load_csv(self, _b):
        path = self._csv_input.value.strip()
        if not path:
            self._load_out.clear_output()
            with self._load_out:
                print('Please enter a path to the summary CSV.')
            return
        self.load_csv(path)

    def _on_view_mode_change(self, change):
        if change['new'] == '2d':
            self._container_2d.layout.display = ''
            self._container_3d.layout.display = 'none'
        else:
            self._container_2d.layout.display = 'none'
            self._container_3d.layout.display = ''
            self._image_stats.value = ''

    def _update_image_stats(self, image_name: str | None) -> None:
        """Render per-image statistics from the CSV into the inline stats panel."""
        if image_name is None or self._summary_df is None:
            self._image_stats.value = ''
            return

        rows = self._summary_df[self._summary_df['image'] == image_name]
        if rows.empty:
            self._image_stats.value = ''
            return

        row = rows.iloc[0]

        # Core activity stats
        n_cells = int(row['n_instances']) if 'n_instances' in row.index and pd.notna(row['n_instances']) else None
        ratio = float(row['activity_ratio']) if 'activity_ratio' in row.index and pd.notna(row['activity_ratio']) else None

        if n_cells is not None and ratio is not None:
            n_active = round(ratio * n_cells)
            n_inactive = n_cells - n_active
            core = (
                f'<b>Cells:</b> {n_cells} &nbsp;|&nbsp; '
                f'<b>Active:</b> {n_active} ({ratio * 100:.1f}%) &nbsp;|&nbsp; '
                f'<b>Inactive:</b> {n_inactive} ({(1 - ratio) * 100:.1f}%)'
            )
        elif n_cells is not None:
            core = f'<b>Cells:</b> {n_cells}'
        elif ratio is not None:
            core = f'<b>Activity ratio:</b> {ratio * 100:.1f}%'
        else:
            core = '<i>No activity data available</i>'

        # Optional metadata fields
        meta_parts: list[str] = []
        for col, label in [('sample', 'Sample'), ('ID', 'ID'), ('z_index', 'Z'), ('time', 'Time'), ('threshold', 'Threshold')]:
            if col in row.index and pd.notna(row[col]):
                val = row[col]
                meta_parts.append(f'<b>{label}:</b> {val:.3g}' if isinstance(val, float) else f'<b>{label}:</b> {val}')
        meta = ' &nbsp;|&nbsp; '.join(meta_parts)

        lines = [core]
        if meta:
            lines.append(meta)

        self._image_stats.value = (
            '<div style="background:#f5f5f5; padding:6px 12px; '
            'border-left:3px solid #5c85d6; font-size:0.88em; margin:4px 0;">'
            + '<br>'.join(lines)
            + '</div>'
        )

    def _on_image_select(self, change):
        if change['new']:
            self._selected_image[0] = change['new']
            self._update_image_stats(change['new'])

    def _update_sample_info(self, _change=None):
        self._sample_info_out.clear_output()
        if self._view_mode.value != '3d':
            return
        sample_id = self._sample_selector.value
        if not sample_id or sample_id == 'No samples available':
            return
        df = self._summary_df
        if df is None or 'ID' not in df.columns:
            return
        with self._sample_info_out:
            rows = df[df['ID'] == sample_id]
            if rows.empty:
                print(f'No data for sample {sample_id}')
                return
            n_slices = len(rows)
            total_cells = (
                int(rows['n_instances'].sum()) if 'n_instances' in rows.columns else 'N/A'
            )
            avg_activity = (
                rows['activity_ratio'].mean() * 100
                if 'activity_ratio' in rows.columns
                else None
            )
            print(f'Sample {sample_id}:')
            print(f'  Z-slices: {n_slices}')
            print(f'  Total cells: {total_cells}')
            if avg_activity is not None:
                print(f'  Average activity rate: {avg_activity:.1f}%')

    def _on_open(self, _b):
        self._popup_out.clear_output()
        with self._popup_out:
            if not NAPARI_AVAILABLE:
                print('Napari is not available. Install with: pip install napari[all]')
                return
            if self._summary_df is None:
                print('Please load a summary CSV first.')
                return
            if self._view_mode.value == '2d':
                self._open_2d()
            else:
                self._open_3d()

    def _on_close(self, _b):
        self._popup_out.clear_output()
        with self._popup_out:
            self._napari_viewer.close()
            print('Napari viewer closed.')

    def _on_show_multiple(self, _b):
        self._popup_out.clear_output()
        with self._popup_out:
            if not NAPARI_AVAILABLE:
                print('Napari is not available. Install with: pip install napari[all]')
                return
            if self._summary_df is None:
                print('Please load a summary CSV first.')
                return
            sample_id = self._sample_selector.value
            if not sample_id or sample_id == 'No samples available':
                print('Please select a sample first.')
                return
            print(f'Opening sample images for {sample_id}...')
            filtered = filter_images_by_sample_id_and_z_index(
                self._summary_df,
                sample_id=sample_id,
                min_z=1,
                max_z=self._max_z.value,
            )
            if filtered.empty:
                print(f'No images found for sample {sample_id}.')
                return

            sample_rows = filtered.sort_values('z_index').head(5)
            viewers = []
            for i, (_, row) in enumerate(sample_rows.iterrows(), start=1):
                selected = row.get('image')
                if not selected:
                    continue
                img = _load_optional(row.get('mcherry_path'), 'mCherry image')
                if img is None:
                    print(f"Skipping '{selected}': mCherry image could not be loaded.")
                    continue

                lbl = _load_optional(row.get('label_path'), 'label image') if self._show_labels.value else None
                activity = _load_optional(row.get('activity_path'), 'activity labels') if self._show_activity.value else None
                bf = _load_optional(row.get('brightfield_path'), 'brightfield image') if self._show_bf.value else None
                activity_is_bin = '_bin' in str(row.get('activity_path', ''))

                viewer = NapariImageViewer()
                viewer.open(
                    img,
                    lbl=lbl,
                    activity_labels=activity,
                    brightfield=bf,
                    image_name=Path(str(selected)).stem,
                    activity_is_bin=activity_is_bin,
                )
                if viewer.current_viewer is not None:
                    viewer.current_viewer.window.move(50 * (i - 1), 50 * (i - 1))
                    viewers.append(viewer.current_viewer)
            if viewers:
                print(f'Opened {len(viewers)} napari viewers.')
            else:
                print('No sample images were opened.')

    # ------------------------------------------------------------------
    # Napari open helpers
    # ------------------------------------------------------------------

    def _open_2d(self):
        selected = self._selected_image[0]
        if not selected:
            print('Please select an image from the list.')
            return

        df = self._summary_df
        assert df is not None
        rows = df[df['image'] == selected]
        if rows.empty:
            print(f"'{selected}' not found in the loaded CSV.")
            return

        row = rows.iloc[0]
        print(f'Opening: {selected}')

        img = _load_optional(row.get('mcherry_path'), 'mCherry image')
        if img is None:
            print('Could not load mCherry image. Check the mcherry_path column in the CSV.')
            return

        lbl = _load_optional(row.get('label_path'), 'label image') if self._show_labels.value else None
        activity = _load_optional(row.get('activity_path'), 'activity labels') if self._show_activity.value else None
        bf = _load_optional(row.get('brightfield_path'), 'brightfield image') if self._show_bf.value else None
        activity_is_bin = '_bin' in str(row.get('activity_path', ''))

        print('Launching napari viewer (2D)...')
        self._napari_viewer.open(
            img,
            lbl=lbl,
            activity_labels=activity,
            brightfield=bf,
            image_name=Path(selected).stem,
            activity_is_bin=activity_is_bin,
        )
        print('Napari viewer opened successfully.')

    def _open_3d(self):
        sample_id = self._sample_selector.value
        if not sample_id or sample_id == 'No samples available':
            print('No valid sample selected.')
            return

        df = self._summary_df
        assert df is not None
        if 'ID' not in df.columns:
            print("Cannot open 3D view: 'ID' column is missing. "
                  "Reload CSV or provide a file_handler.")
            return

        filtered = filter_images_by_sample_id_and_z_index(df, sample_id=sample_id, min_z=1, max_z=self._max_z.value)

        if filtered.empty:
            print(f"No rows found for sample '{sample_id}'.")
            return

        print(f'Building 3D volume for sample {sample_id} ({len(filtered)} z-slices)...')

        def _build_volume(col: str, label: str) -> np.ndarray | None:
            slices = [_load_optional(row.get(col), label) for _, row in filtered.iterrows()]
            valid = [s for s in slices if s is not None]
            if not valid:
                return None
            ref_shape = valid[0].shape
            ref_dtype = valid[0].dtype
            filled = [
                s if s is not None else np.zeros(ref_shape, dtype=ref_dtype)
                for s in slices
            ]
            if len(valid) < len(slices):
                print(f'  Warning: {len(slices) - len(valid)} missing {label} slice(s) filled with zeros.')
            return np.stack(filled, axis=0)

        mcherry_vol = _build_volume('mcherry_path', 'mCherry')
        if mcherry_vol is None:
            print(f'No mCherry images could be loaded for sample {sample_id}.')
            return

        label_vol = _build_volume('label_path', 'labels') if self._show_labels.value else None
        activity_vol = _build_volume('activity_path', 'activity') if self._show_activity.value else None
        bf_vol = _build_volume('brightfield_path', 'brightfield') if self._show_bf.value else None

        activity_is_bin = bool(
            filtered['activity_path'].dropna()
            .apply(lambda p: '_bin' in str(p))
            .any()
        )

        print('Launching napari viewer (3D)...')
        self._napari_viewer.open_3d(
            image_volume=mcherry_vol,
            label_volume=label_vol,
            activity_volume=activity_vol,
            brightfield_volume=bf_vol,
            title=f'3D Z-Stack: Sample {sample_id}',
            show_original=self._show_mcherry.value,
            show_labels=self._show_labels.value,
            show_activity=self._show_activity.value,
            show_brightfield=self._show_bf.value,
            activity_is_bin=activity_is_bin,
        )
        print('Napari 3D viewer opened successfully.')

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_csv(self, csv_path: str | Path) -> bool:
        """
        Load an activity summary CSV and populate the image and sample lists.

        If the CSV lacks an ``ID`` column and a ``file_handler`` was provided
        at construction, sample IDs are derived from the ``mcherry_path``
        column via ``file_handler.extract_unique_id()``.

        Parameters
        ----------
        csv_path : str or Path
            Path to the ``_summary.csv`` file.

        Returns
        -------
        bool
            True on success, False on error.
        """
        self._load_out.clear_output()
        try:
            p = Path(str(csv_path)).expanduser()
            if not p.exists():
                with self._load_out:
                    print(f'File not found: {p}')
                return False

            df = pd.read_csv(p)

            # Derive ID column from mcherry_path if absent and file_handler available
            file_handler = self.file_handler
            if 'ID' not in df.columns and file_handler is not None:
                df = add_meta_info(df, file_handler=file_handler)

            self._summary_df = df

            # Populate 2D image list
            if 'image' in df.columns:
                self._image_list.options = df['image'].tolist()
                if self._image_list.options:
                    first = self._image_list.options[0]
                    self._selected_image[0] = first
                    self._update_image_stats(first)

            # Populate 3D sample list
            if 'ID' in df.columns:
                samples = sorted(df['ID'].dropna().unique().tolist())
                self._sample_selector.options = samples if samples else ['No samples available']
            else:
                self._sample_selector.options = ['No samples available']

            self._update_sample_info()

            with self._load_out:
                print(f'Loaded {len(df)} rows from {p.name}')
                if 'ID' in df.columns:
                    print(f'Samples found: {df["ID"].nunique()}')

            return True

        except Exception as e:
            with self._load_out:
                print(f'Error loading CSV: {e}')
                import traceback
                traceback.print_exc()
            return False

    def display(self):
        """Render the widget in a Jupyter / Voila cell."""
        ipy_display(self.widget)


def create_and_display_napari_viewer(
    csv_path: str | Path | None = None,
    file_handler=None,
) -> ImageViewer:
    """
    Create and optionally initialize the ipywidget-based napari launcher.

    Parameters
    ----------
    csv_path : str or Path, optional
        Summary CSV to load immediately before displaying the widget.
    file_handler : optional
        Passed through to ``ImageViewer`` for deriving sample IDs when needed.

    Returns
    -------
    ImageViewer
        The initialized widget controller.
    """
    viewer = ImageViewer(file_handler=file_handler)
    if csv_path is not None:
        viewer.load_csv(csv_path)
    viewer.display()
    return viewer
